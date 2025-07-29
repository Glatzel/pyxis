use std::sync::Arc;
use std::thread;
use std::time::Duration;

use proj::Context;

pub fn init_proj_builder() -> miette::Result<Arc<Context>> {
    let ctx = proj::Context::new();
    ctx.set_log_level(proj::LogLevel::Trace)?;
    Ok(ctx)
}
#[cfg(debug_assertions)]
pub fn start_deadlock_detection() {
    tokio::task::spawn_blocking(|| {
        loop {
            thread::sleep(Duration::from_secs(10));
            let deadlocks = parking_lot::deadlock::check_deadlock();
            if deadlocks.is_empty() {
                continue;
            }

            clerk::error!("{} deadlocks detected", deadlocks.len());
            for (i, threads) in deadlocks.iter().enumerate() {
                let mut msg = format!("Deadlock #{}\n", i);
                for t in threads {
                    msg.push_str(format!("Thread Id {:#?}\n", t.thread_id()).as_str());
                    msg.push_str(format!("{:#?}", t.backtrace()).as_str());
                }
                clerk::error!("{}", msg);
            }
        }
    });
}
