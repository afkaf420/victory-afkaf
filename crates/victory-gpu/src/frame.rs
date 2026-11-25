//! Frame pacing: the title sets a frame interval and waits for each frame
//! to begin. Ours is a clock started by the first interval set, ticking
//! once per interval; the token is the frame number. Each event kind
//! (origin, vblank, flip) happens once per tick, so two waits for the same
//! kind never end on the same tick: the title polls the origin
//! with "token on time-out" (starting its next frame at once) and is paced
//! by a blocking wait for the flip before each present.
//!
//! shortcut: event offsets are ignored (every event is at the tick); tie
//! the clock to presentation if a title paces by them.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::object::{HRESULT, S_FALSE, S_OK, This};

/// What a wait returns when its time is up first.
const TIMEOUT: HRESULT = S_FALSE;
/// The frame-origin event: the one that starts frames and makes tokens.
const ORIGIN: u32 = 0;
/// Make the frame's token even if the wait times out (the title starts the
/// frame early).
const TOKEN_ON_TIMEOUT: u32 = 2;
const SIXTIETH: Duration = Duration::from_micros(16_667);
const MIN_TARGET_FPS: u32 = 1;
const MAX_TARGET_FPS: u32 = 1_000;

/// Opt-in frame-pacing overrides controlled by `VICTORY_UNLOCK_FPS=1` and
/// `VICTORY_TARGET_FPS=<N>` (1..=1000).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PacingConfig {
    /// When true, enable VRR/tearing presentation (`DXGI_PRESENT_ALLOW_TEARING`
    /// when supported by host DXGI) and bypass artificial console frame-interval
    /// blocking unless `target_fps` requests an explicit cap.
    pub unlock_fps: bool,
    /// Optional host frame-rate cap (1..=1000 FPS). Overrides slower title
    /// intervals such as 30 Hz (33,333 us) with `1_000_000_000 / fps` ns.
    pub target_fps: Option<u32>,
}

impl PacingConfig {
    /// Parses `VICTORY_UNLOCK_FPS` and `VICTORY_TARGET_FPS` values.
    pub fn parse(unlock: Option<&str>, target: Option<&str>) -> Self {
        let unlock_fps = unlock.is_some_and(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true")
        });
        let target_fps = target
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&fps| (MIN_TARGET_FPS..=MAX_TARGET_FPS).contains(&fps));
        Self {
            unlock_fps,
            target_fps,
        }
    }

    /// True when frame waits should return immediately without sleeping on the
    /// synthetic interval clock (fully uncapped / host-VRR paced).
    pub const fn is_uncapped(self) -> bool {
        self.unlock_fps && self.target_fps.is_none()
    }

    /// Target interval implied by `VICTORY_TARGET_FPS`, if set.
    pub fn target_interval(self) -> Option<Duration> {
        self.target_fps
            .map(|fps| Duration::from_nanos(1_000_000_000 / u64::from(fps)))
    }

    /// Effective interval applied when the title calls `SetFrameIntervalX`.
    pub fn effective_interval(self, requested: Duration) -> Duration {
        match self.target_interval() {
            Some(target) => requested.min(target),
            None => requested,
        }
    }
}

fn config() -> PacingConfig {
    static CONFIG: std::sync::OnceLock<PacingConfig> = std::sync::OnceLock::new();
    *CONFIG.get_or_init(|| {
        let unlock = std::env::var("VICTORY_UNLOCK_FPS").ok();
        let target = std::env::var("VICTORY_TARGET_FPS").ok();
        let cfg = PacingConfig::parse(unlock.as_deref(), target.as_deref());
        if cfg.unlock_fps || cfg.target_fps.is_some() {
            tracing::info!(
                unlock_fps = cfg.unlock_fps,
                target_fps = ?cfg.target_fps,
                "frame pacing override active"
            );
        }
        cfg
    })
}

/// Whether `VICTORY_UNLOCK_FPS=1` is enabled (enables VRR/tearing presentation
/// when supported by host DXGI).
pub fn allow_tearing() -> bool {
    config().unlock_fps
}

/// Configured `VICTORY_TARGET_FPS` cap, if any.
pub fn target_fps() -> Option<u32> {
    config().target_fps
}

struct Clock {
    start: Instant,
    interval: Duration,
    /// The tick number at `start` (ticks go on across interval changes).
    first: u64,
}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);
static FRAME: AtomicU64 = AtomicU64::new(0);

fn clock() -> std::sync::MutexGuard<'static, Option<Clock>> {
    CLOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method: a frame every `length` microseconds. The period is how
/// many intervals a frame may take to reach the display (pipeline depth),
/// not part of the rate.
pub(crate) extern "system" fn set_interval(
    _: This,
    _output: This,
    length: u32,
    _period: u32,
    _flags: u32,
) -> HRESULT {
    let requested = match length {
        0 => SIXTIETH,
        us => Duration::from_micros(u64::from(us)),
    };
    let cfg = config();
    let interval = cfg.effective_interval(requested);
    tracing::info!(
        ?requested,
        ?interval,
        uncapped = cfg.is_uncapped(),
        "frame interval"
    );
    // A new interval starts from now, after the last tick handed out.
    let first = LAST
        .iter()
        .map(|t| t.load(Ordering::Relaxed))
        .max()
        .unwrap_or(0);
    *clock() = Some(Clock {
        start: Instant::now(),
        interval,
        first,
    });
    S_OK
}

/// Device method: events are scheduled at every tick.
pub(crate) extern "system" fn schedule(
    _: This,
    _kind: u32,
    _offset: u32,
    _list: *const u8,
    _flags: u32,
) -> HRESULT {
    S_OK
}

/// Frames begun so far (frame waits that ended).
pub(crate) fn current() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

/// Per event kind (origin, vblank, flip and later kinds): the last tick a
/// wait for it ended on.
static LAST: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn last(kind: u32) -> &'static AtomicU64 {
    &LAST[(kind as usize).min(LAST.len() - 1)]
}

/// The next tick after now and after the last one a wait for `kind` ended
/// on (an event happens once per tick: a wait woken a little early must not
/// let the next one end on the same tick), and how long until it.
fn next_tick(kind: u32, cfg: PacingConfig) -> (u64, Duration) {
    let mut clock = clock();
    let c = clock.get_or_insert_with(|| Clock {
        start: Instant::now(),
        interval: cfg.effective_interval(SIXTIETH),
        first: 0,
    });
    let elapsed = c.start.elapsed().as_nanos();
    let interval = c.interval.as_nanos().max(1);
    let tick =
        (c.first + (elapsed / interval + 1) as u64).max(last(kind).load(Ordering::Relaxed) + 1);
    let at = u128::from(tick - c.first) * interval;
    (
        tick,
        Duration::from_nanos(at.saturating_sub(elapsed) as u64),
    )
}

/// Sleeps for `dur`, yielding the final tail when pacing at high frame rates
/// so coarse OS timer ticks do not drag a 120+ FPS cap down.
fn sleep_for(dur: Duration, high_precision: bool) {
    if dur.is_zero() {
        return;
    }
    if !high_precision {
        std::thread::sleep(dur);
        return;
    }
    let deadline = Instant::now() + dur;
    let tail = Duration::from_millis(2);
    if dur > tail {
        std::thread::sleep(dur - tail);
    }
    while Instant::now() < deadline {
        std::thread::yield_now();
    }
}

/// Device method: blocks until the next tick (or the time-out); an origin
/// wait then hands out the new frame's token. Other events keep the token
/// the title passes. Ticks are absolute (start plus whole intervals), so
/// lateness never accumulates.
pub(crate) extern "system" fn wait(
    _: This,
    kind: u32,
    timeout_ms: u32,
    _list: *const u8,
    flags: u32,
    token: *mut u64,
) -> HRESULT {
    wait_with_config(config(), kind, timeout_ms, flags, token)
}

fn wait_with_config(
    cfg: PacingConfig,
    kind: u32,
    timeout_ms: u32,
    flags: u32,
    token: *mut u64,
) -> HRESULT {
    let origin = kind == ORIGIN;
    let (hr, tick, wait) = if cfg.is_uncapped() {
        // Unlocked / VRR mode: never artificially sleep on a console interval;
        // advance the event tick immediately and let the host swap chain pace.
        let tick = last(kind).fetch_add(1, Ordering::Relaxed) + 1;
        (S_OK, tick, Duration::ZERO)
    } else {
        let (mut tick, mut wait) = next_tick(kind, cfg);
        let limit = Duration::from_millis(u64::from(timeout_ms));
        // A blocking wait for an event that has happened since the last wait
        // for it ended (a frame took longer than a tick) ends at once, on the
        // tick that passed: sleeping on to the next one cost 17 ms of every
        // 730 ms frame (run 58: "2.0 frame waits sleeping 16.8 ms"). Polls (a
        // time-out shorter than a tick) answer as before.
        // `VICTORY_LATE_WAITS=1` sleeps to the next tick, as before.
        static SLEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*SLEEP.get_or_init(|| std::env::var_os("VICTORY_LATE_WAITS").is_some_and(|v| v == "1"))
            && limit >= wait
            && tick > last(kind).load(Ordering::Relaxed) + 1
        {
            tick -= 1;
            wait = Duration::ZERO;
        }
        let high_precision = cfg.target_fps.is_some_and(|fps| fps >= 60);
        let hr = if wait > limit {
            sleep_for(limit, false);
            TIMEOUT
        } else {
            sleep_for(wait, high_precision);
            last(kind).fetch_max(tick, Ordering::Relaxed);
            S_OK
        };
        (hr, tick, wait.min(limit))
    };
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        use crate::copystat::{C, add};
        add(C::PaceWaits, 1);
        add(C::PaceUs, wait.as_micros() as u64);
    }
    // A frame begun on a time-out takes no tick: its flip still does.
    if origin && (hr == S_OK || flags & TOKEN_ON_TIMEOUT != 0) {
        // Tokens count frames (consecutive), not ticks.
        let frame = FRAME.fetch_add(1, Ordering::Relaxed) + 1;
        if !token.is_null() {
            // SAFETY: checked non-null; the title passes its token.
            unsafe { token.write(frame) };
        }
    }
    #[cfg(feature = "diagnostics")]
    tracing::debug!(
        kind,
        timeout_ms,
        flags,
        hr,
        tick,
        wait_us = wait.as_micros() as u64,
        frame = current(),
        "frame wait"
    );
    let _ = (tick, wait);
    hr
}

/// Device method: no statistics are kept, so none are returned.
pub(crate) extern "system" fn statistics(
    _: This,
    _token: u64,
    _kinds: u32,
    count: *mut u32,
    _out: *mut u8,
) -> HRESULT {
    if !count.is_null() {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
    }
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn pacing_config_parses_unlock_and_target_fps() {
        let default = PacingConfig::parse(None, None);
        assert!(!default.unlock_fps);
        assert_eq!(default.target_fps, None);
        assert!(!default.is_uncapped());
        assert_eq!(
            default.effective_interval(Duration::from_micros(33_333)),
            Duration::from_micros(33_333)
        );

        let uncapped = PacingConfig::parse(Some("1"), None);
        assert!(uncapped.unlock_fps);
        assert_eq!(uncapped.target_fps, None);
        assert!(uncapped.is_uncapped());

        let capped = PacingConfig::parse(Some("true"), Some("120"));
        assert!(capped.unlock_fps);
        assert_eq!(capped.target_fps, Some(120));
        assert!(!capped.is_uncapped());
        assert_eq!(
            capped.effective_interval(Duration::from_micros(33_333)),
            Duration::from_nanos(1_000_000_000 / 120)
        );
        // Never slows down a faster requested interval.
        assert_eq!(
            capped.effective_interval(Duration::from_micros(4_000)),
            Duration::from_micros(4_000)
        );

        let invalid = PacingConfig::parse(Some("0"), Some("5000"));
        assert!(!invalid.unlock_fps);
        assert_eq!(invalid.target_fps, None);
    }

    #[test]
    fn unlocked_wait_returns_immediately_without_30hz_blocking() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let null = std::ptr::null();
        assert_eq!(set_interval(null, null, 33_333, 3, 0), S_OK);
        let uncapped = PacingConfig {
            unlock_fps: true,
            target_fps: None,
        };
        const FLIP: u32 = 2;
        let before = Instant::now();
        let mut prev_token = 0u64;
        for i in 0..8 {
            let mut token = 0u64;
            assert_eq!(
                wait_with_config(uncapped, ORIGIN, 0, TOKEN_ON_TIMEOUT, &mut token),
                S_OK
            );
            if i > 0 {
                assert_eq!(token, prev_token + 1);
            }
            prev_token = token;
            assert_eq!(
                wait_with_config(uncapped, FLIP, u32::MAX, 0, &mut token),
                S_OK
            );
        }
        // Eight 30 Hz frames would take ~266 ms if blocked; uncapped is instant.
        assert!(before.elapsed() < Duration::from_millis(50));
    }

    #[test]
    fn waits_end_on_ticks_and_count_frames() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let null = std::ptr::null();
        assert_eq!(set_interval(null, null, 2_000, 1, 0), S_OK);
        let mut first = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut first), S_OK);
        let before = Instant::now();
        let mut second = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut second), S_OK);
        assert!(second > first);
        assert!(before.elapsed() <= Duration::from_millis(50));
        assert_eq!(wait(null, 0, 0, std::ptr::null(), 0, &mut second), TIMEOUT);
        // Tokens count frames, a frame started early on a time-out too.
        let mut early = 0;
        let flags = TOKEN_ON_TIMEOUT;
        assert_eq!(
            wait(null, 0, 0, std::ptr::null(), flags, &mut early),
            TIMEOUT
        );
        let mut next = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut next), S_OK);
        assert!(early == second + 1 && next == early + 1);
        assert!(LAST[0].load(Ordering::Relaxed) >= 2);

        // The title's loop (the clock is global, so in the same test): an
        // origin poll that times out (and starts the frame), then a
        // blocking flip wait. Each flip takes its own tick.
        const FLIP: u32 = 2;
        assert_eq!(set_interval(null, null, 5_000, 3, 0), S_OK);
        let mut token = 0;
        let flips: Vec<u64> = (0..4)
            .map(|_| {
                let _ = wait(null, 0, 0, std::ptr::null(), TOKEN_ON_TIMEOUT, &mut token);
                assert_eq!(
                    wait(null, FLIP, u32::MAX, std::ptr::null(), 0, &mut token),
                    S_OK
                );
                LAST[2].load(Ordering::Relaxed)
            })
            .collect();
        assert!(flips.windows(2).all(|w| w[1] == w[0] + 1), "{flips:?}");
        // Even when a flip wait begins on the tick another just ended on.
        let (tick, _) = next_tick(FLIP, config());
        assert!(tick > flips[3]);
    }
}
