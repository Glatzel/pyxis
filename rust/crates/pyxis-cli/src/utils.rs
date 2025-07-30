use std::sync::Arc;

use proj::Context;
#[cfg(debug_assertions)]
use tokio::sync::watch;

pub fn init_proj_builder() -> miette::Result<Arc<Context>> {
    let ctx = proj::Context::new();
    ctx.set_log_level(proj::LogLevel::Trace)?;
    Ok(ctx)
}
#[cfg(debug_assertions)]
pub fn start_deadlock_detection(shutdown_rx: watch::Receiver<()>) {
    tokio::task::spawn_blocking(move || {
        loop {
            if shutdown_rx.has_changed().is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(10));
            let deadlocks = parking_lot::deadlock::check_deadlock();
            if deadlocks.is_empty() {
                continue;
            }

            clerk::error!("{} deadlocks detected", deadlocks.len());
            for (i, threads) in deadlocks.iter().enumerate() {
                let mut msg = format!("Deadlock #{i}\n");
                for t in threads {
                    msg.push_str(format!("Thread Id {:#?}\n", t.thread_id()).as_str());
                    msg.push_str(format!("{:#?}", t.backtrace()).as_str());
                }
                clerk::error!("{}", msg);
            }
        }
    });
}
