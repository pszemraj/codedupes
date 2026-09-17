import assert from "node:assert/strict";
import test from "node:test";

import { fetchRecords, readSnapshot } from "../src/pagination.mjs";

const implementations = [fetchRecords, readSnapshot];

function transport(pages, beforeResolve = () => {}) {
  const calls = [];
  return {
    calls,
    async fetch(cursor) {
      calls.push(cursor);
      assert.ok(pages.has(cursor), `Unexpected request: ${cursor}`);
      const value = pages.get(cursor);
      beforeResolve(cursor);
      if (value instanceof Error) throw value;
      return value;
    },
  };
}

for (const implementation of implementations) {
  test(`${implementation.name}: stable first-occurrence results and trace`, async () => {
    const pages = new Map([
      [
        null,
        Object.freeze({
          rows: Object.freeze([
            Object.freeze({ id: "a", value: 1 }),
            Object.freeze({ id: "b", value: 2 }),
          ]),
          next: "next",
        }),
      ],
      [
        "next",
        Object.freeze({
          rows: Object.freeze([
            Object.freeze({ id: "a", value: 99 }),
            Object.freeze({ id: "c", value: 3 }),
          ]),
          next: null,
        }),
      ],
    ]);
    const mock = transport(pages);
    assert.deepEqual(await implementation(mock.fetch), {
      records: [
        { id: "a", value: 1 },
        { id: "b", value: 2 },
        { id: "c", value: 3 },
      ],
      pages: 2,
    });
    assert.deepEqual(mock.calls, [null, "next"]);
  });

  test(`${implementation.name}: empty intermediate page is followed`, async () => {
    const mock = transport(
      new Map([
        [null, { rows: [], next: "x" }],
        ["x", { rows: [], next: null }],
      ]),
    );
    assert.deepEqual(await implementation(mock.fetch), { records: [], pages: 2 });
    assert.deepEqual(mock.calls, [null, "x"]);
  });

  test(`${implementation.name}: budget stops before an extra request`, async () => {
    const mock = transport(new Map([[null, { rows: [], next: "x" }]]));
    await assert.rejects(implementation(mock.fetch, { maxPages: 1 }), {
      name: "RangeError",
      message: "PAGE_LIMIT",
    });
    assert.deepEqual(mock.calls, [null]);
  });

  test(`${implementation.name}: exactly-at-budget terminal success`, async () => {
    const mock = transport(new Map([[null, { rows: [], next: null }]]));
    assert.deepEqual(await implementation(mock.fetch, { maxPages: 1 }), {
      records: [],
      pages: 1,
    });
  });

  test(`${implementation.name}: cursor cycle wins over exhausted budget`, async () => {
    const mock = transport(
      new Map([
        [null, { rows: [], next: "x" }],
        ["x", { rows: [], next: "x" }],
      ]),
    );
    await assert.rejects(implementation(mock.fetch, { maxPages: 2 }), {
      name: "Error",
      message: "CURSOR_CYCLE",
    });
    assert.deepEqual(mock.calls, [null, "x"]);
  });

  test(`${implementation.name}: already-aborted signal makes no requests`, async () => {
    const controller = new AbortController();
    controller.abort();
    const mock = transport(new Map());
    await assert.rejects(implementation(mock.fetch, { signal: controller.signal }), {
      message: "ABORTED",
    });
    assert.deepEqual(mock.calls, []);
  });

  test(`${implementation.name}: cancellation during transport discards response`, async () => {
    const controller = new AbortController();
    const mock = transport(
      new Map([[null, { rows: [], next: "x" }]]),
      () => controller.abort(),
    );
    await assert.rejects(implementation(mock.fetch, { signal: controller.signal }), {
      message: "ABORTED",
    });
    assert.deepEqual(mock.calls, [null]);
  });

  test(`${implementation.name}: immediate post-call abort does not create an extra request`, async () => {
    const controller = new AbortController();
    const mock = transport(new Map([[null, { rows: [], next: "x" }]]));
    const pending = implementation(mock.fetch, { signal: controller.signal });
    assert.deepEqual(mock.calls, [null]);
    controller.abort();
    await assert.rejects(pending, { message: "ABORTED" });
    assert.deepEqual(mock.calls, [null]);
  });

  test(`${implementation.name}: malformed page/cursor/row rejects`, async () => {
    for (const [page, message] of [
      [null, "INVALID_PAGE"],
      [{ rows: new Array(1), next: null }, "INVALID_ROW"],
      [{ rows: null, next: null }, "INVALID_PAGE"],
      [{ rows: [], next: "" }, "INVALID_CURSOR"],
      [{ rows: [] }, "INVALID_CURSOR"],
      [{ rows: [{ id: "a", value: Number.NaN }], next: null }, "INVALID_ROW"],
      [{ rows: [{ id: "", value: 1 }], next: null }, "INVALID_ROW"],
      [
        {
          rows: [
            { id: "a", value: 1 },
            { id: "a", value: 1.5 },
          ],
          next: null,
        },
        "INVALID_ROW",
      ],
    ]) {
      const mock = transport(new Map([[null, page]]));
      await assert.rejects(implementation(mock.fetch), { name: "TypeError", message });
    }
  });

  test(`${implementation.name}: invalid limits reject before calling transport`, async () => {
    for (const maxPages of [0, -1, 1.5, Number.POSITIVE_INFINITY, Number.NaN]) {
      const mock = transport(new Map());
      await assert.rejects(implementation(mock.fetch, { maxPages }), {
        name: "RangeError",
        message: "INVALID_PAGE_LIMIT",
      });
      assert.deepEqual(mock.calls, []);
    }
  });

  test(`${implementation.name}: preserves the transport rejection object`, async () => {
    const error = new Error("offline failure");
    const mock = transport(new Map([[null, error]]));
    await assert.rejects(implementation(mock.fetch), (received) => received === error);
  });

  test(`${implementation.name}: synchronous transport throws become rejections`, async () => {
    const error = new Error("immediate failure");
    await assert.rejects(
      implementation(() => {
        throw error;
      }),
      (received) => received === error,
    );
  });
}

test("generated page chains agree with an independently assembled oracle and request order", async () => {
  for (let length = 1; length <= 100; length += 1) {
    const pages = new Map();
    const expected = [];
    for (let index = 0; index < length; index += 1) {
      const key = index === 0 ? null : `page-${index}`;
      const row = { id: `item-${index}`, value: index - 50 };
      const rows = index === 0 ? [row] : [row, { id: "item-0", value: 10000 }];
      pages.set(key, {
        rows,
        next: index + 1 === length ? null : `page-${index + 1}`,
      });
      expected.push(row);
    }
    for (const implementation of implementations) {
      const mock = transport(pages);
      assert.deepEqual(await implementation(mock.fetch), {
        records: expected,
        pages: length,
      });
      assert.deepEqual(mock.calls, Array.from(pages.keys()));
    }
  }
});
