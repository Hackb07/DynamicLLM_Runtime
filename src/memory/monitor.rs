use sysinfo::{System, SystemExt};

pub struct MemoryMonitor {
    system: System,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        Self { system }
    }

    pub fn get_memory_info(&mut self) -> (u64, u64) {
        self.system.refresh_memory();
        let total = self.system.total_memory() / 1024; // MB
        let used = self.system.used_memory() / 1024;
        (total, used)
    }

    pub fn get_available_memory_mb(&mut self) -> u64 {
        self.system.refresh_memory();
        self.system.available_memory() / 1024
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}
