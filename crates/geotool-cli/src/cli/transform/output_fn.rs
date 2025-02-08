use super::record::Record;

pub fn output_simple(record: &Record) {
    println!(
        "{}:{}, {}:{}, {}:{}\n",
        record.ox_name, record.ox, record.oy_name, record.oy, record.oz_name, record.oz,
    )
}
pub fn output_plain(records: &[Record]) {
    println!("Transform Records");
    println!("=================");
    if let Some(input) = records.first() {
        println!(
            "{}:{}, {}:{}, {}:{}",
            input.ox_name, input.ox, input.oy_name, input.oy, input.oz_name, input.oz,
        )
    }
    for record in records.iter().skip(1) {
        println!(
            r#"    step:{}
    method: {}
    from: {}
    to: {}
{}:{}, {}:{}, {}:{}"#,
            record.idx,
            record.method,
            record.from,
            record.to,
            record.ox_name,
            record.ox,
            record.oy_name,
            record.oy,
            record.oz_name,
            record.oz,
        )
    }
    println!();
}
pub fn output_json(records: &[Record]) {
    let json_txt = serde_json::to_string_pretty(records).unwrap();
    println!("{json_txt}\n")
}
