import init, { cowsay, fixture_version } from "./pkg/cowsay_dupe_fixture.js";

const elements = {
  controls: document.querySelector("#controls"),
  message: document.querySelector("#message"),
  width: document.querySelector("#width"),
  widthValue: document.querySelector("#width-value"),
  thinking: document.querySelector("#thinking"),
  wrapper: document.querySelector("#wrapper"),
  output: document.querySelector("#output"),
  status: document.querySelector("#status"),
  version: document.querySelector("#version"),
};

function render() {
  const width = Number.parseInt(elements.width.value, 10);
  const useFoldWrapper = elements.wrapper.value === "fold";
  elements.widthValue.value = String(width);
  elements.output.textContent = cowsay(
    elements.message.value,
    width,
    elements.thinking.checked,
    useFoldWrapper,
  );
}

async function main() {
  await init();
  elements.status.textContent = "Rust/Wasm ready";
  elements.version.textContent = `crate v${fixture_version()}`;
  elements.controls.addEventListener("input", render);
  elements.controls.addEventListener("change", render);
  render();
}

main().catch((error) => {
  console.error(error);
  elements.status.textContent = "WebAssembly failed to load";
  elements.output.textContent = String(error);
});
