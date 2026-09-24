// Run with: node --test tests/test_dashboard.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const source = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
    .map(match => match[1]).join('\n');

function dashboard() {
    const elements = new Map();
    const plots = [];
    function element() {
        return {
            textContent: '', innerHTML: '', style: {}, children: [], listeners: {},
            appendChild(child) { this.children.push(child); },
            addEventListener(type, listener) { this.listeners[type] = listener; },
        };
    }
    function get(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
    }
    get('ctrl-precision').options = ['bf16', 'fp8', 'int8', 'fp32']
        .map(value => ({ value, disabled: false }));
    const context = vm.createContext({
        window: { addEventListener() {} },
        document: { getElementById: get, createElement: element, querySelectorAll() { return []; } },
        Plotly: { newPlot(id, traces) { plots.push({ id, traces }); } },
    });
    const run = code => vm.runInContext(code, context);
    run(source);
    return { get, run, plots };
}

test('8B prefill converts billions correctly and preserves OOM output', () => {
    const app = dashboard();
    app.run('renderCapabilities(HARDWARE_DB.apple_m4)');
    const rows = app.get('llm-matrix-body').children;
    assert.match(rows[0].innerHTML, /1905 ms/);
    assert.match(rows[3].innerHTML, /OOM on Single Device/);
    assert.doesNotMatch(rows[3].innerHTML, /\d+ ms/);
});

test('SRAM keeps FP32 accumulators across input precisions and flags overflow', () => {
    const app = dashboard();
    for (const [precision, expected] of [['bf16', '128.0 KB'], ['fp8', '96.0 KB'],
                                        ['int8', '96.0 KB'], ['fp32', '192.0 KB']]) {
        app.run(`currentChipKey = 'nvidia_h100_sxm'; currentPrecision = '${precision}'; renderTileSimulation()`);
        assert.equal(app.get('tile-sram-req').textContent, expected);
    }
    app.run("currentChipKey = 'apple_a17_pro'; currentPrecision = 'bf16'; currentTileM = 160; renderTileSimulation()");
    assert.equal(app.get('tile-sram-req').textContent, '152.0 KB');
    assert.equal(app.get('tier-sram-status').textContent, 'CRITICAL OVERFLOW!');
});

test('precision change updates the curve, operating point, and SRAM', () => {
    const app = dashboard();
    app.run("currentChipKey = 'nvidia_h100_sxm'; currentIntensity = 1000; setupEventListeners()");
    for (const [precision, peak] of [['bf16', 989], ['fp8', 1978], ['int8', 1978], ['fp32', 67]]) {
        app.get('ctrl-precision').listeners.change({ target: { value: precision } });
        const traces = app.plots.at(-1).traces;
        assert.equal(traces[0].y.at(-1), peak);
        assert.equal(traces[1].y[0], peak);
        assert.match(traces[0].name, new RegExp(precision.toUpperCase()));
        assert.ok(app.get('tile-sram-req').textContent);
    }
});

test('chip changes disable unsupported precision and select a supported fallback', () => {
    const app = dashboard();
    app.run("currentChipKey = 'apple_m4'; currentPrecision = 'fp8'; updatePrecisionOptions()");
    assert.equal(app.get('ctrl-precision').value, 'bf16');
    assert.equal(app.get('ctrl-precision').options.find(option => option.value === 'fp8').disabled, true);
    app.run(`for (const key of Object.keys(HARDWARE_DB)) {
        currentChipKey = key;
        updatePrecisionOptions();
        renderRooflineChart();
    }`);
    for (const plot of app.plots) {
        assert.ok(plot.traces[0].y.every(value => Number.isFinite(value) && value > 0));
    }
});
