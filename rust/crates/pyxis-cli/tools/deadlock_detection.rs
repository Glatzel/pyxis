#[tokio::main]
async fn main() ->Result<(),ProjError> {
    pyxis_cli::logging::init_log(clerk::LogLevel::TRACE);
    start_deadlock_detection();
    pyxis_cli::cli::execute().await?;
    Ok(())
}
pub fn start_deadlock_detection() {
    tokio::task::spawn_blocking(|| {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3));
            let deadlocks = parking_lot::deadlock::check_deadlock();
            if deadlocks.is_empty() {
                continue;
            }

            clerk::error!("{} deadlocks detected", deadlocks.len());
            if let Some((i, threads)) = deadlocks.iter().enumerate().next() {
                let mut msg = format!("Deadlock #{i}\n");
                for t in threads {
                    msg.push_str(format!("Thread Id {:#?}\n", t.thread_id()).as_str());
                    msg.push_str(format!("{:#?}", t.backtrace()).as_str());
                }
                clerk::error!("{}", msg);
                std::process::exit(1);
            }
        }
    });
}
