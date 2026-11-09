// Medidas citadas na METODOLOGIA.md (seção 4.6.1). Não é teste: imprime tabelas, com semente fixa.
//   node tests/estudo_dirigido_medidas.js
"use strict";
const path = require("path");
const M = require(path.resolve(__dirname, "..", "estudo_dirigido_motor.js"));

// ---------------------------------------------------------------------------
// Parte 1: neurônio morto (ReLU) ou saturado (tanh, logística) com pesos sorteados
// ---------------------------------------------------------------------------
let seed = 12345;
const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
const sorteio = () => Math.round((rnd() * 2 - 1) * 100) / 100;       // [-1, 1], 2 casas (igual à página)

function percentualRuim(modo, xmax, n, nin) {
  let ruim = 0;
  for (let k = 0; k < n; k++) {
    const W = [[Array.from({ length: nin }, sorteio)]], b = [[sorteio()]];
    const r = M.criarRede([nin, 1], modo, W, b);
    const x = Array.from({ length: nin }, () => Math.round(rnd() * xmax * 2) / 2);   // 0 a xmax, passo 0,5
    if (!M.avancar(r, x).viva[0][0]) ruim++;
  }
  return (100 * ruim / n).toFixed(1);
}

console.log("1) Experimento 1 (2 entradas), pesos e viés sorteados em [-1, 1], 20.000 sorteios por célula");
console.log("   % de neurônios mortos (ReLU) ou saturados (tanh, logística)");
console.log("   x máx |  ReLU |  tanh | logística");
for (const xm of [1, 4, 20]) {
  console.log("  ", String(xm).padStart(5), "|", percentualRuim("relu", xm, 20000, 2).padStart(5), "|",
    percentualRuim("tanh", xm, 20000, 2).padStart(5), "|", percentualRuim("sigmoid", xm, 20000, 2).padStart(9));
}

console.log("\n2) Faixa de z em que o neurônio ainda conta como vivo (derivada ≥ 10% do máximo)");
for (const modo of ["tanh", "sigmoid"]) {
  const r = M.criarRede([1, 1], modo, [[[1]]], [[0]]);
  let lo = null, hi = null;
  for (let z = -10; z <= 10; z += 0.01) {
    if (M.avancar(r, [z]).viva[0][0]) { if (lo === null) lo = z; hi = z; }
  }
  console.log(`   ${modo}: z entre ${lo.toFixed(2)} e ${hi.toFixed(2)}`);
}

console.log("\n3) Detector 2×2 com ReLU, imagem diagonal, pesos sorteados");
let mortos = 0;
const N = 20000;
for (let k = 0; k < N; k++) {
  const W = [[Array.from({ length: 4 }, sorteio)]], b = [[sorteio()]];
  if (!M.avancar(M.criarRede([4, 1], "relu", W, b), [1, 0, 0, 1]).viva[0][0]) mortos++;
}
console.log(`   morto em ${(100 * mortos / N).toFixed(0)}% dos ${N} sorteios`);

// ---------------------------------------------------------------------------
// Parte 2: rede de duas camadas da captura de tela (tanh nas duas), alvo no limite da faixa
// ---------------------------------------------------------------------------
function rede(oculta, saida) {
  return M.criarRede([2, 2, 1], saida,
    [[[0.044, 0.515], [0.342, -0.83]], [[-0.05, 0.922]]],
    [[-0.86, -0.11], [1.2]], oculta);
}
function passosAte(r, x, y, lr, perdaAlvo, teto) {
  for (let k = 0; k <= teto; k++) {
    if (M.perdaDoLote(r, [{ x, y: [y] }]) < perdaAlvo) return k;
    M.aplicarPasso(r, [{ x, y: [y] }], lr);
  }
  return ">" + teto;
}
const x = [1, -1];
const g = M.retropropagar(rede("tanh", "tanh"), x, [1]);
console.log("\n4) Rede 2-2-1 (x = [1, -1], η = 0,2). Estado da captura de tela, alvo y = 1");
console.log(`   saída a = ${g.frente.saida[0].toFixed(3)}, z = ${g.frente.z[1][0].toFixed(2)}, tanh'(z) = ${g.frente.deriv[1][0].toFixed(3)}, perda = ${g.perda.toFixed(4)}`);
console.log(`   δ_saída = ${g.delta[1][0].toExponential(1)}, δ_A = ${g.delta[0][0].toExponential(1)}, δ_B = ${g.delta[0][1].toExponential(1)}`);
console.log("   Passos para a perda ficar abaixo de 1e-4:");
console.log("   - tanh/tanh, alvo +1 (no limite da faixa):", passosAte(rede("tanh", "tanh"), x, 1, 0.2, 1e-4, 200000));
console.log("   Passos para a perda ficar abaixo de 0,01 (mesmos pesos):");
console.log("   - tanh/tanh, alvo −1 (lado oposto), η 0,2:", passosAte(rede("tanh", "tanh"), x, -1, 0.2, 0.01, 200000));
console.log("   - tanh/tanh, alvo −1 (lado oposto), η 0,5:", passosAte(rede("tanh", "tanh"), x, -1, 0.5, 0.01, 200000));
console.log("   - ReLU/linear, alvo −1:                  ", passosAte(rede("relu", "linear"), x, -1, 0.2, 0.01, 200000));
console.log("   - logística/logística, alvo 0 (limite):  ", passosAte(rede("sigmoid", "sigmoid"), x, 0, 0.2, 0.01, 200000));

// ---------------------------------------------------------------------------
// Parte 3: o alvo no limite da faixa é mais lento que o alvo dentro dela?
// Mesmos pesos iniciais da captura, tanh/tanh, η = 0,2. Passos para a perda ficar abaixo de 1e-4.
// ---------------------------------------------------------------------------
console.log("\n5) Mesmos pesos, tanh/tanh, η = 0,2: passos até perda < 1e-4, conforme o alvo");
for (const y of [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]) {
  const r = rede("tanh", "tanh");
  const p0 = M.perdaDoLote(r, [{ x, y: [y] }]);
  console.log(`   alvo ${y.toFixed(2)}: perda inicial ${p0.toFixed(4)} -> ${passosAte(r, x, y, 0.2, 1e-4, 200000)} passos`);
}
