use super::record::Record;

pub fn output_simple(record: &Record) {
    println!(
        "{}:{}, {}:{}, {}:{}",
        record.ox_name, record.ox, record.oy_name, record.oy, record.oz_name, record.oz,
    )
}
pub fn output_plain(records: &[Record]) {
    println!("Transform Records");
    println!("=================");
    for record in records {
        println!(
            r#"
    step:{}
    method: {}
    from: {}
    to: {}
{}:{}, {}:{}, {}:{}
"#,
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
}
pub fn output_json(records: &[Record]) {
    let json_txt = serde_json::to_string_pretty(records).unwrap();
    println!("{json_txt}")
}
