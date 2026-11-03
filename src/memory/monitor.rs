use crate::memory::estimator::{ResidencyInfo, ResidencyTracker};
use serde::{Deserialize, Serialize};

/// Point-in-time host memory readings, in megabytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub total_mb: Option<u64>,
    pub available_mb: Option<u64>,
    pub used_mb: Option<u64>,
}

/// Host memory plus per-model residency.
///
/// Measured-vs-assumed behavior: `host` is MEASURED from the OS when
/// the platform supports it (`None` fields otherwise). Each entry in
/// `models` carries both an ASSUMED estimate (always present, from
/// config heuristics) and an optional MEASURED footprint (present only
/// when a backend reported one). `resident_mb_effective` mixes the two
/// via [`ResidencyInfo::effective_mb`]; it is an observability aid, not
/// the admission budget (which still uses lifecycle assumed sizes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencySnapshot {
    pub host: MemorySnapshot,
    pub models: Vec<ResidencyInfo>,
    pub resident_mb_effective: u64,
}

/// Dependency-free host memory monitor.
///
/// Windows reads `GlobalMemoryStatusEx`; Linux parses `/proc/meminfo`;
/// other platforms report `None` and callers proceed without a host gate.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryMonitor;

impl MemoryMonitor {
    pub fn new() -> Self {
        Self
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        let (total_mb, available_mb) = match system_memory_mb() {
            Some((total, available)) => (Some(total), Some(available)),
            None => (None, None),
        };
        let used_mb = match (total_mb, available_mb) {
            (Some(total), Some(available)) => Some(total.saturating_sub(available)),
            _ => None,
        };
        MemorySnapshot {
            total_mb,
            available_mb,
            used_mb,
        }
    }

    pub fn total_memory_mb(&self) -> Option<u64> {
        self.snapshot().total_mb
    }

    pub fn available_memory_mb(&self) -> Option<u64> {
        self.snapshot().available_mb
    }

    /// Combine a host snapshot with per-model residency from the
    /// tracker's eviction hooks. Stateless: the tracker owns residency,
    /// the monitor only joins it with host readings.
    pub fn residency_snapshot(&self, tracker: &ResidencyTracker) -> ResidencySnapshot {
        ResidencySnapshot {
            host: self.snapshot(),
            models: tracker.snapshot(),
            resident_mb_effective: tracker.resident_mb(),
        }
    }
}

#[cfg(windows)]
fn system_memory_mb() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 {
        return None;
    }
    Some((
        (status.ullTotalPhys / 1024 / 1024) as u64,
        (status.ullAvailPhys / 1024 / 1024) as u64,
    ))
}

#[cfg(target_os = "linux")]
fn system_memory_mb() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = None;
    let mut available_kb = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = rest.split_whitespace().next()?.parse::<u64>().ok();
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available_kb = rest.split_whitespace().next()?.parse::<u64>().ok();
        }
    }
    Some((total_kb? / 1024, available_kb? / 1024))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn system_memory_mb() -> Option<(u64, u64)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_is_internally_consistent() {
        let snapshot = MemoryMonitor::new().snapshot();
        match (snapshot.total_mb, snapshot.available_mb, snapshot.used_mb) {
            (Some(total), Some(available), Some(used)) => {
                assert!(total > 0);
                assert!(available <= total);
                assert_eq!(used, total - available);
            }
            (None, None, None) => {
                // Unsupported platform: all unknown, never partial.
            }
            other => panic!("partial memory snapshot: {other:?}"),
        }
    }
}
