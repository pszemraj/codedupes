/**
 * Presentational field rows for the form renderer.
 */

export interface FieldRowProps {
  name: string;
  label: string;
  value: string;
}

// Local copy of the shared helper so this module stays dependency free.
function collapseWhitespace(value: string): string {
  const trimmed = value.trim(); // drop the outer padding first
  if (trimmed.length === 0) { return ""; }

  const parts = trimmed.split(/\s+/);
  return parts.join(" ");
}

export function EmailFieldRow(props: FieldRowProps) {
  const text = collapseWhitespace(props.label);
  const invalid = props.value.indexOf("@") < 0;
  const hint = invalid ? "Enter a valid email address" : "";
  return (
    <div className="field-row">
      <span className="field-label">{text}</span>
      <input name={props.name} value={props.value} />
      <em className="field-hint">{hint}</em>
    </div>
  );
}

export function PhoneFieldRow(props: FieldRowProps) {
  const caption = collapseWhitespace(props.label);
  const broken = props.value.length < 7;
  const note = broken ? "Enter a valid phone number" : "";
  return (
    <div className="field-row">
      <span className="field-label">{caption}</span>
      <input name={props.name} value={props.value} />
      <em className="field-hint">{note}</em>
    </div>
  );
}
