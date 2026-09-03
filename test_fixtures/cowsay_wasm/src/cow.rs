pub(crate) fn render(thinking: bool) -> String {
    let connector = if thinking { 'o' } else { '\\' };

    [
        format!("        {connector}   ^__^"),
        format!("         {connector}  (oo)\\_______"),
        "            (__)\\       )\\/\\".to_owned(),
        "                ||----w |".to_owned(),
        "                ||     ||".to_owned(),
    ]
    .join("\n")
}
