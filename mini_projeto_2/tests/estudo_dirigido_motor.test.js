// Testes do motor do estudo dirigido. Rode com:  node tests/estudo_dirigido_motor.test.js
"use strict";
const assert = require("assert");
const M = require("../estudo_dirigido_motor.js");

let falhas = 0;
function teste(nome, fn) {
  try { fn(); console.log("ok   -", nome); }
  catch (e) { falhas++; console.log("FALHA-", nome, "\n       ", e.message); }
}
const perto = (a, b, tol = 1e-6) => assert.ok(Math.abs(a - b) <= tol, `${a} != ${b} (tol ${tol})`);

// Imagens 2x2 do experimento do Jupyter: [p1,p2,p3,p4] -> alvo
const DIAG = { x: [1, 0, 0, 1], y: [1] };
const HORIZ = { x: [1, 1, 0, 0], y: [0] };
const PRETA = { x: [0, 0, 0, 0], y: [0] };
const ANTI = { x: [0, 1, 1, 0], y: [0] };
const QUATRO = [DIAG, HORIZ, PRETA, ANTI];

const detector = (w, b) => M.criarRede([4, 1], "relu", [[w]], [[b]]);

teste("pesos do Jupyter reproduzem as predições da seção 4.5", () => {
  const r = detector([0.315, -0.320, 0.038, 0.674], 0.006);
  perto(M.avancar(r, DIAG.x).saida[0], 0.995);
  perto(M.avancar(r, HORIZ.x).saida[0], 0.001);
  perto(M.avancar(r, PRETA.x).saida[0], 0.006);
  perto(M.avancar(r, ANTI.x).saida[0], 0, 1e-12); // z = -0,276 -> ReLU zera
});

teste("Dying ReLU: w=0,1 e b=-0,5 na diagonal dá z=-0,3, saída 0 e gradiente exatamente zero", () => {
  const r = detector([0.1, 0.1, 0.1, 0.1], -0.5);
  const f = M.avancar(r, DIAG.x);
  perto(f.z[0][0], -0.3);
  assert.strictEqual(f.saida[0], 0);
  assert.strictEqual(f.viva[0][0], false);
  const g = M.retropropagar(r, DIAG.x, DIAG.y);
  assert.deepStrictEqual(g.gW[0][0], [0, 0, 0, 0]);
  assert.deepStrictEqual(g.gb[0], [0]);
  perto(g.perda, 0.5); // o erro existe (saída 0, alvo 1), mas não chega aos pesos
});

teste("Dying ReLU: o passo de SGD não muda nenhum peso, mesmo repetido 100 vezes", () => {
  const r = detector([0.1, 0.1, 0.1, 0.1], -0.5);
  const antes = JSON.stringify([r.W, r.b]);
  for (let i = 0; i < 100; i++) {
    const info = M.aplicarPasso(r, [DIAG], 0.1);
    assert.strictEqual(info.mudou, false);
  }
  assert.strictEqual(JSON.stringify([r.W, r.b]), antes);
});

teste("com b=0 o mesmo neurônio sai do zero e a perda cai (quebra de simetria da 4.5)", () => {
  const r = detector([0.1, 0.1, 0.1, 0.1], 0.0);
  const perda0 = M.perdaDoLote(r, QUATRO);
  for (let i = 0; i < 100; i++) M.aplicarPasso(r, QUATRO, 0.1);
  const perda1 = M.perdaDoLote(r, QUATRO);
  assert.ok(perda1 < perda0 * 0.1, `perda ${perda0} -> ${perda1}`);
  assert.ok(r.W[0][0][1] < 0, "o peso do pixel 2 deve ficar inibitório (negativo)");
});

teste("gradiente analítico bate com diferenças finitas (rede 2-2-1, saída linear)", () => {
  const r = M.criarRede([2, 2, 1], "linear", [[[1, 2], [3, 0]], [[2, 1]]], [[0.5, -0.2], [0.1]]);
  const x = [2, 1], y = [10];
  const g = M.retropropagar(r, x, y);
  const h = 1e-6;
  for (let l = 0; l < r.W.length; l++) {
    for (let i = 0; i < r.W[l].length; i++) {
      for (let j = 0; j < r.W[l][i].length; j++) {
        const r1 = M.clonar(r), r2 = M.clonar(r);
        r1.W[l][i][j] += h; r2.W[l][i][j] -= h;
        const num = (M.perdaDoLote(r1, [{ x, y }]) - M.perdaDoLote(r2, [{ x, y }])) / (2 * h);
        perto(g.gW[l][i][j], num, 1e-5);
      }
      const r1 = M.clonar(r), r2 = M.clonar(r);
      r1.b[l][i] += h; r2.b[l][i] -= h;
      perto(g.gb[l][i], (M.perdaDoLote(r1, [{ x, y }]) - M.perdaDoLote(r2, [{ x, y }])) / (2 * h), 1e-5);
    }
  }
});

teste("neurônio oculto com z<=0 bloqueia a culpa: δ_oculto = (W2ᵀ·δ_saída) ⊙ ReLU'(z1)", () => {
  // Neurônio B tem z_B = -4 (morto); neurônio A está vivo.
  const r = M.criarRede([2, 2, 1], "linear", [[[1, 2], [-3, 0]], [[2, 1]]], [[0, 0], [0]]);
  const g = M.retropropagar(r, [2, 1], [10]);
  assert.strictEqual(g.frente.viva[0][0], true);
  assert.strictEqual(g.frente.viva[0][1], false);
  assert.strictEqual(g.delta[0][1], 0);             // sem a derivada seria W2[0][1]*δ_out ≠ 0
  assert.notStrictEqual(g.delta[0][0], 0);
  assert.deepStrictEqual(g.gW[0][1], [0, 0]);       // nenhum peso de B recebe gradiente
});

teste("valores do exemplo da página: rede 2-2-1 com x=[2,1], alvo 10 dá saída 14, δ_out=4", () => {
  const r = M.criarRede([2, 2, 1], "linear", [[[1, 2], [3, 0]], [[2, 1]]], [[0, 0], [0]]);
  const g = M.retropropagar(r, [2, 1], [10]);
  perto(g.frente.saida[0], 2 * 4 + 1 * 6);          // z_A=4, z_B=6 -> 2*4+1*6 = 14
  perto(g.delta[1][0], 4);                          // 14 - 10
});

teste("perceptron da página (w=[-1,5; 1; 0,5], x=[1,2,1]) calcula z, δ e gradientes corretos", () => {
  const r = M.criarRede([3, 1], "relu", [[[-1.5, 1, 0.5]]], [[0]]);
  const g = M.retropropagar(r, [1, 2, 1], [2]);
  perto(g.frente.z[0][0], -1.5 + 2 + 0.5);           // 1,0
  perto(g.delta[0][0], 1 - 2);                        // a - y = -1 (ReLU viva)
  assert.deepStrictEqual(g.gW[0][0].map((v) => +v.toFixed(6)), [-1, -2, -1]);
});

teste("passo de SGD com várias amostras usa a média dos gradientes", () => {
  const r = detector([0.2, 0.2, 0.2, 0.2], 0.5);
  const so1 = M.gradienteDoLote(r, [DIAG]);
  const so2 = M.gradienteDoLote(r, [HORIZ]);
  const ambos = M.gradienteDoLote(r, [DIAG, HORIZ]);
  perto(ambos.gb[0][0], (so1.gb[0][0] + so2.gb[0][0]) / 2);
  perto(ambos.perda, (so1.perda + so2.perda) / 2);
});

// ---------------------------------------------------------------------------
// Outras ativações: tanh e logística
// ---------------------------------------------------------------------------
const detectorAtiv = (modo, w, b) => M.criarRede([4, 1], modo, [[w]], [[b]]);

teste("valores de referência: tanh(0)=0, logística(0)=0,5, f'(0) = 1 e 0,25", () => {
  perto(M.ATIVACOES.tanh.f(0), 0);
  perto(M.ATIVACOES.sigmoid.f(0), 0.5);
  perto(M.ATIVACOES.tanh.df(0), 1);
  perto(M.ATIVACOES.sigmoid.df(0), 0.25);
  perto(M.ATIVACOES.sigmoid.f(2), 1 / (1 + Math.exp(-2)));
  perto(M.ATIVACOES.sigmoid.f(-800), 0, 1e-12);       // não estoura
  perto(M.ATIVACOES.sigmoid.f(800), 1, 1e-12);
});

teste("cada camada usa a sua ativação: oculta tanh e saída logística", () => {
  const r = M.criarRede([2, 2, 1], "sigmoid", null, null, "tanh");
  assert.strictEqual(M.ativacaoDaCamada(r, 0), "tanh");
  assert.strictEqual(M.ativacaoDaCamada(r, 1), "sigmoid");
  const c = M.clonar(r);
  assert.strictEqual(c.ativacaoOculta, "tanh");
  assert.strictEqual(c.ativacaoSaida, "sigmoid");
});

teste("gradiente bate com diferenças finitas: oculta tanh, saída logística", () => {
  const r = M.criarRede([2, 3, 1], "sigmoid", [[[0.5, -0.3], [0.2, 0.8], [-0.6, 0.1]], [[0.7, -0.4, 0.9]]], [[0.1, -0.2, 0.05], [0.3]], "tanh");
  const x = [1.5, -0.5], y = [0.8];
  const g = M.retropropagar(r, x, y);
  const h = 1e-6;
  for (let l = 0; l < r.W.length; l++) for (let i = 0; i < r.W[l].length; i++) {
    for (let j = 0; j < r.W[l][i].length; j++) {
      const r1 = M.clonar(r), r2 = M.clonar(r);
      r1.W[l][i][j] += h; r2.W[l][i][j] -= h;
      perto(g.gW[l][i][j], (M.perdaDoLote(r1, [{ x, y }]) - M.perdaDoLote(r2, [{ x, y }])) / (2 * h), 1e-6);
    }
    const r1 = M.clonar(r), r2 = M.clonar(r);
    r1.b[l][i] += h; r2.b[l][i] -= h;
    perto(g.gb[l][i], (M.perdaDoLote(r1, [{ x, y }]) - M.perdaDoLote(r2, [{ x, y }])) / (2 * h), 1e-6);
  }
});

teste("gradiente bate com diferenças finitas: oculta logística, saída tanh", () => {
  const r = M.criarRede([2, 2, 1], "tanh", [[[0.4, -0.7], [0.9, 0.2]], [[-0.5, 0.6]]], [[0.0, 0.1], [-0.1]], "sigmoid");
  const x = [0.3, 1.2], y = [0.4];
  const g = M.retropropagar(r, x, y);
  const h = 1e-6;
  for (let l = 0; l < r.W.length; l++) for (let i = 0; i < r.W[l].length; i++) for (let j = 0; j < r.W[l][i].length; j++) {
    const r1 = M.clonar(r), r2 = M.clonar(r);
    r1.W[l][i][j] += h; r2.W[l][i][j] -= h;
    perto(g.gW[l][i][j], (M.perdaDoLote(r1, [{ x, y }]) - M.perdaDoLote(r2, [{ x, y }])) / (2 * h), 1e-6);
  }
});

teste("saturação: logística com z=5 e tanh com z=2 contam como saturadas; z=0 não", () => {
  const sig = (z) => M.avancar(M.criarRede([1, 1], "sigmoid", [[[1]]], [[0]]), [z]).viva[0][0];
  const tnh = (z) => M.avancar(M.criarRede([1, 1], "tanh", [[[1]]], [[0]]), [z]).viva[0][0];
  assert.strictEqual(sig(0), true);
  assert.strictEqual(sig(2), true);
  assert.strictEqual(sig(5), false);
  assert.strictEqual(tnh(0), true);
  assert.strictEqual(tnh(1), true);
  assert.strictEqual(tnh(2), false);
});

teste("diferença central: saturada na logística o gradiente é minúsculo mas NÃO é zero; na ReLU morta é zero", () => {
  const sat = detectorAtiv("sigmoid", [0.1, 0.1, 0.1, 0.1], 10);       // z = 10 na imagem preta
  const gs = M.retropropagar(sat, PRETA.x, [1]);                      // alvo 1 mas a saída já está perto de 1... use alvo 0
  const gs0 = M.retropropagar(sat, PRETA.x, [0]);
  assert.strictEqual(gs0.frente.viva[0][0], false);
  assert.ok(gs0.gb[0][0] > 0 && gs0.gb[0][0] < 1e-3, `gb = ${gs0.gb[0][0]}`);
  const morta = detectorAtiv("relu", [0.1, 0.1, 0.1, 0.1], -10);
  assert.strictEqual(M.retropropagar(morta, DIAG.x, DIAG.y).gb[0][0], 0);
  assert.ok(gs.perda >= 0);
});

teste("logística não 'morre' com w=0,1 e b=-0,5 (z=-0,3): o treino muda os pesos e a perda cai", () => {
  const r = detectorAtiv("sigmoid", [0.1, 0.1, 0.1, 0.1], -0.5);
  const p0 = M.perdaDoLote(r, QUATRO);
  const antes = JSON.stringify([r.W, r.b]);
  let mudou = false;
  for (let i = 0; i < 200; i++) mudou = M.aplicarPasso(r, QUATRO, 0.5).mudou || mudou;
  assert.strictEqual(mudou, true);
  assert.notStrictEqual(JSON.stringify([r.W, r.b]), antes);
  assert.ok(M.perdaDoLote(r, QUATRO) < p0, `${p0} -> ${M.perdaDoLote(r, QUATRO)}`);
});

teste("tanh com alvo 1 na diagonal: a perda cai e a saída fica entre −1 e 1", () => {
  const r = detectorAtiv("tanh", [0.1, 0.1, 0.1, 0.1], 0);
  const p0 = M.perdaDoLote(r, [DIAG]);
  for (let i = 0; i < 200; i++) M.aplicarPasso(r, [DIAG], 0.3);
  const a = M.avancar(r, DIAG.x).saida[0];
  assert.ok(M.perdaDoLote(r, [DIAG]) < p0 * 0.1);
  assert.ok(a > 0.8 && a < 1);
});

teste("alvo fora da faixa da logística nunca é alcançado (y=2): a perda para em ½(1−2)²", () => {
  const r = M.criarRede([1, 1], "sigmoid", [[[0]]], [[0]]);
  for (let i = 0; i < 5000; i++) M.aplicarPasso(r, [{ x: [1], y: [2] }], 2);
  const a = M.avancar(r, [1]).saida[0];
  assert.ok(a < 1 && a > 0.99);
  perto(M.perdaDoLote(r, [{ x: [1], y: [2] }]), 0.5, 0.01);
});

console.log(falhas ? `\n${falhas} teste(s) falharam` : "\ntodos os testes passaram");
process.exit(falhas ? 1 : 0);
