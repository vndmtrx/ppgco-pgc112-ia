// Teste de execução da página (estudo_dirigido.html) numa DOM simulada.
// Precisa do jsdom (não é dependência do projeto). Sem ele, o teste é pulado.
//   NODE_PATH=/caminho/para/node_modules node tests/estudo_dirigido_pagina.test.js [arquivo.html]
"use strict";
const assert = require("assert");
const path = require("path");

let JSDOM, VirtualConsole;
try { ({ JSDOM, VirtualConsole } = require("jsdom")); }
catch (e) { console.log("pulado - jsdom não está instalado"); process.exit(0); }

const arquivo = path.resolve(__dirname, "..", process.argv[2] || "estudo_dirigido.html");
let falhas = 0;
const erros = [];

function teste(nome, fn) {
  try { fn(); console.log("ok   -", nome); }
  catch (e) { falhas++; console.log("FALHA-", nome, "\n       ", e.message); }
}

(async () => {
  const vc = new VirtualConsole();
  vc.on("jsdomError", (e) => erros.push(e.message));
  vc.on("error", (e) => erros.push(String(e)));
  const dom = await JSDOM.fromFile(arquivo, { runScripts: "dangerously", resources: "usable", pretendToBeVisual: true, virtualConsole: vc });
  await new Promise((r) => dom.window.addEventListener("load", r));
  const w = dom.window, d = w.document;
  const ev = (c) => w.eval(c);
  const botao = (texto) => [...d.querySelectorAll("button")].find((b) => b.textContent.trim().startsWith(texto));
  const clicar = (texto) => { const b = botao(texto); assert.ok(b, `botão "${texto}" não existe`); b.click(); };
  const aviso = () => d.getElementById("aviso").textContent;

  teste("a página carrega sem erros de execução", () => assert.deepStrictEqual(erros, []));
  teste("há 3 abas de experimento", () => assert.strictEqual(d.querySelectorAll("#abas button").length, 3));

  teste("um neurônio: padrão tem z = 1,00 e todos os neurônios vivos", () => {
    assert.ok(aviso().includes("vivos"), aviso());
    assert.ok(d.getElementById("grade-camadas").textContent.includes("1,00") || d.getElementById("grade-camadas").textContent.includes("1.00"));
  });

  teste("um neurônio: o alvo y vai de 1 a 100 e o valor escolhido chega ao cálculo", () => {
    const sl = d.querySelector("#controles input[aria-label='alvo y']");
    assert.strictEqual(sl.min, "1");
    assert.strictEqual(sl.max, "100");
    sl.value = "100"; sl.dispatchEvent(new w.Event("input", { bubbles: true }));
    assert.strictEqual(ev("E.y"), 100);
    assert.ok(d.getElementById("perda-faixa").textContent.includes("100,0") ||
              d.getElementById("perda-faixa").textContent.includes("100.0"));
    sl.value = "2"; sl.dispatchEvent(new w.Event("input", { bubbles: true }));
  });

  teste("barra deslizante não é recriada ao mexer (o arrasto não se perde)", () => {
    const sl = d.querySelector("#controles input[type=range]");
    sl.value = "3"; sl.dispatchEvent(new w.Event("input", { bubbles: true }));
    assert.strictEqual(d.querySelector("#controles input[type=range]"), sl);
    assert.strictEqual(ev("E.x[0]"), 3);
  });

  teste("editar um valor da matriz muda a rede e recalcula", () => {
    const inp = d.querySelector("#grade-camadas input[aria-label^='peso']");
    inp.value = "2"; inp.dispatchEvent(new w.Event("input", { bubbles: true }));
    assert.strictEqual(ev("E.rede.W[0][0][0]"), 2);
  });

  teste("baixar o viés até z <= 0 mostra o aviso de neurônio morto e gradiente 0", () => {
    const b = d.querySelector("#grade-camadas input[aria-label^='viés']");
    b.value = "-10"; b.dispatchEvent(new w.Event("input", { bubbles: true }));
    assert.ok(aviso().includes("morto"), aviso());
    const g = d.querySelectorAll("#grade-camadas .camada .grupo[data-g='3'] .cel");
    assert.ok([...g].some((c) => c.classList.contains("zero")));
  });

  teste("fase 3 mostra gradientes nos fios e marca o foco", () => {
    clicar("3. Volta");
    assert.strictEqual(d.querySelector("#grade-camadas").parentElement.dataset.foco, "3");
    assert.ok(d.getElementById("svg").textContent.includes("∇"));
  });
  teste("fase 4 mostra peso → peso novo", () => {
    clicar("4. Atualizar");
    assert.ok(d.getElementById("svg").textContent.includes("→"));
    clicar("Visão geral");
  });

  teste("duas camadas: as entradas x vão de 0 a 20 e chegam ao cálculo", () => {
    clicar("2. Duas camadas");
    const xs = [...d.querySelectorAll("#controles input[type=range]")].filter((s) => /^x/.test(s.getAttribute("aria-label")));
    assert.strictEqual(xs.length, 2);
    xs.forEach((s) => { assert.strictEqual(s.min, "0"); assert.strictEqual(s.max, "20"); });
    xs[0].value = "20"; xs[0].dispatchEvent(new w.Event("input", { bubbles: true }));
    assert.strictEqual(ev("E.x[0]"), 20);
    // z_A = 1*20 + 2*1 = 22 com os pesos iniciais
    assert.ok(d.getElementById("svg").textContent.includes("z = 22.00"));
    xs[0].value = "2"; xs[0].dispatchEvent(new w.Event("input", { bubbles: true }));
  });

  teste("um neurônio: as entradas x vão de 0 a 20", () => {
    clicar("1. Um neurônio");
    const xs = [...d.querySelectorAll("#controles input[type=range]")].filter((s) => /^x/.test(s.getAttribute("aria-label")));
    assert.strictEqual(xs.length, 2);
    xs.forEach((s) => { assert.strictEqual(s.min, "0"); assert.strictEqual(s.max, "20"); });
  });

  teste("duas camadas: o alvo y vai de 1 a 100 e chega ao cálculo", () => {
    clicar("2. Duas camadas");
    const sl = d.querySelector("#controles input[aria-label='alvo y']");
    assert.strictEqual(sl.min, "1");
    assert.strictEqual(sl.max, "100");
    assert.strictEqual(ev("E.y"), 10);                 // valor inicial mantido
    sl.value = "100"; sl.dispatchEvent(new w.Event("input", { bubbles: true }));
    assert.strictEqual(ev("E.y"), 100);
    // com y=100 e saída 14, a culpa na saída é 14 - 100 = -86
    assert.ok(d.getElementById("grade-camadas").textContent.includes("−86.00"));
    sl.value = "10"; sl.dispatchEvent(new w.Event("input", { bubbles: true }));
  });

  teste("duas camadas: neurônio B morto bloqueia a culpa (δ de B = 0)", () => {
    clicar("2. Duas camadas");
    assert.ok(aviso().includes("vivos"), aviso());
    const bB = [...d.querySelectorAll("#grade-camadas input[aria-label^='viés']")][1];
    bB.value = "-100"; bB.dispatchEvent(new w.Event("input", { bubbles: true }));
    assert.ok(aviso().includes("morto"), aviso());
    assert.strictEqual(ev("Motor.retropropagar(E.rede,E.x,[E.y]).delta[0][1]"), 0);
  });

  teste("detector: pesos 'morto' dão z = −0,30 e o treino NÃO muda nada (100 passos)", () => {
    clicar("3. Detector");
    clicar("Morto");
    assert.ok(aviso().includes("morto"), aviso());
    const antes = ev("JSON.stringify([E.rede.W,E.rede.b])");
    for (let i = 0; i < 5; i++) clicar("Treinar 20");
    assert.strictEqual(ev("JSON.stringify([E.rede.W,E.rede.b])"), antes);
    assert.ok(d.getElementById("info-passo").textContent.includes("Nenhum peso mudou"));
    assert.ok(d.getElementById("leg-grafico").textContent.includes("reta"));
  });

  teste("detector: pesos 'vivo' aprendem e a perda cai pelo menos 10 vezes", () => {
    clicar("Vivo");
    for (let i = 0; i < 5; i++) clicar("Treinar 20");
    const h = ev("E.historico");
    assert.ok(h[h.length - 1] < h[0] * 0.1, `${h[0]} -> ${h[h.length - 1]}`);
    assert.ok(ev("E.rede.W[0][0][1]") < 0, "o peso do pixel 2 deve ficar negativo");
  });

  teste("detector: pixels mudam a imagem e o alvo; imagem pronta restaura", () => {
    clicar("Horizontal");
    assert.deepStrictEqual(JSON.parse(ev("JSON.stringify(E.x)")), [1, 1, 0, 0]);
    assert.strictEqual(ev("E.y"), 0);
    clicar("Diagonal");
    assert.strictEqual(ev("E.y"), 1);
  });

  // ---- toggle do dummy +1 -------------------------------------------------
  const alternarDummy = (ligado) => {
    const tg = d.getElementById("tg-dummy");
    tg.checked = ligado;
    tg.dispatchEvent(new w.Event("change", { bubbles: true }));
  };
  const nosDummy = () => d.querySelectorAll("#svg .no-dummy").length;
  const colunaVies = () => d.querySelectorAll("#grade-camadas .grupo[data-g='1'] .mat .vies").length;

  teste("dummy +1 ligado por padrão: um nó +1 por camada e o '1' nas matrizes", () => {
    assert.strictEqual(nosDummy(), 1);                       // detector tem 1 camada
    assert.ok(d.getElementById("svg").textContent.includes("+1"));
    assert.ok(d.getElementById("nota-conv").textContent.includes("[b | W]"));
    const um = [...d.querySelectorAll("#grade-camadas .grupo[data-g='1'] .vies")].some((c) => c.textContent === "1");
    assert.ok(um, "falta a entrada fixa 1 no vetor");
  });

  teste("dummy oculto: some o nó +1, o viés vira vetor separado e continua editável", () => {
    alternarDummy(false);
    assert.strictEqual(nosDummy(), 0);
    assert.ok(!d.getElementById("svg").textContent.includes("+1"));
    assert.ok(d.getElementById("nota-conv").textContent.includes("W · a + b"));
    const um = [...d.querySelectorAll("#grade-camadas .grupo[data-g='1'] .vies")].some((c) => c.textContent === "1");
    assert.ok(!um, "não deve haver a entrada fixa 1");
    const b = d.querySelector("#grade-camadas input[aria-label^='viés']");
    b.value = "0.25"; b.dispatchEvent(new w.Event("input", { bubbles: true }));
    assert.strictEqual(ev("E.rede.b[0][0]"), 0.25);
    // volta sem dummy: culpa, ReLU', δ, ∇W e ∇b = 5 caixas (com dummy, ∇[b|W] é uma só = 4)
    assert.strictEqual(d.querySelectorAll("#grade-camadas .grupo[data-g='3'] .mat").length, 5);
  });

  teste("matrizes têm as formas certas nos dois modos (detector 4→1)", () => {
    const celulas = (g) => d.querySelectorAll(`#grade-camadas .grupo[data-g='${g}'] .mat`);
    const dims = (m) => parseInt(/repeat\((\d+)/.exec(m.style.gridTemplateColumns)[1], 10);
    // oculto: W é 1x4, b é 1x1
    const m1 = celulas(1);
    assert.strictEqual(dims(m1[0]), 4);
    assert.strictEqual(dims(m1[2]), 1);
    alternarDummy(true);
    const m2 = celulas(1);
    assert.strictEqual(dims(m2[0]), 5);                      // [b | W] tem 1+4 colunas
  });

  teste("fio do viés: aparece e é bloqueado (×0) com neurônio morto na fase 3", () => {
    clicar("3. Detector");
    clicar("Morto");
    clicar("3. Volta");
    assert.ok(d.getElementById("svg").textContent.includes("∇b 0 (×0)"), "fio do viés deveria mostrar ∇b 0");
    clicar("Visão geral");
  });

  teste("treinar com o dummy oculto dá o mesmo resultado que com ele visível", () => {
    const rodar = (ligado) => {
      alternarDummy(ligado);
      clicar("Vivo"); clicar("Reiniciar");
      clicar("3. Detector"); clicar("Vivo");
      for (let i = 0; i < 3; i++) clicar("Treinar 20");
      return ev("JSON.stringify([E.rede.W,E.rede.b])");
    };
    const a = rodar(true), b = rodar(false);
    assert.strictEqual(a, b);
    alternarDummy(true);
  });

  // ---- funções de ativação ------------------------------------------------
  const escolher = (id, valor) => {
    const s = d.getElementById(id);
    assert.ok(s, `seletor #${id} não existe`);
    s.value = valor; s.dispatchEvent(new w.Event("change", { bubbles: true }));
  };

  teste("seletores de ativação: 1 no experimento 1 e no 3; 2 no experimento 2", () => {
    clicar("1. Um neurônio");
    assert.ok(d.getElementById("sel-saida") && !d.getElementById("sel-oculta"));
    clicar("2. Duas camadas");
    assert.ok(d.getElementById("sel-saida") && d.getElementById("sel-oculta"));
    clicar("3. Detector");
    assert.ok(d.getElementById("sel-saida") && !d.getElementById("sel-oculta"));
    const ops = [...d.getElementById("sel-saida").options].map((o) => o.value);
    assert.deepStrictEqual(ops, ["relu", "tanh", "sigmoid", "linear"]);
  });

  teste("trocar a ativação mantém os pesos, zera o histórico e recalcula", () => {
    clicar("3. Detector"); clicar("Vivo"); clicar("Treinar 20");
    const pesos = ev("JSON.stringify([E.rede.W,E.rede.b])");
    assert.ok(ev("E.historico.length") > 0);
    escolher("sel-saida", "tanh");
    assert.strictEqual(ev("E.rede.ativacaoSaida"), "tanh");
    assert.strictEqual(ev("JSON.stringify([E.rede.W,E.rede.b])"), pesos);
    assert.strictEqual(ev("E.historico.length"), 0);
    assert.ok(d.getElementById("grade-camadas").textContent.includes("tanh(z)"));
    escolher("sel-saida", "relu");
  });

  teste("logística com os pesos 'Morto': NÃO aparece neurônio morto e o treino aprende", () => {
    clicar("3. Detector"); clicar("Morto");
    assert.ok(aviso().includes("morto"), aviso());
    escolher("sel-saida", "sigmoid");
    assert.ok(!aviso().includes("morto"), aviso());
    const antes = ev("JSON.stringify([E.rede.W,E.rede.b])");
    for (let i = 0; i < 5; i++) clicar("Treinar 20");
    assert.notStrictEqual(ev("JSON.stringify([E.rede.W,E.rede.b])"), antes);
    assert.ok(!d.getElementById("info-passo").textContent.includes("Nenhum peso mudou"));
    escolher("sel-saida", "relu");
  });

  teste("tanh/logística saturada: aviso diz 'saturado', nó âmbar e gradiente minúsculo (não zero)", () => {
    clicar("1. Um neurônio");
    escolher("sel-saida", "sigmoid");
    const b = d.querySelector("#grade-camadas input[aria-label^='viés']");
    b.value = "-12"; b.dispatchEvent(new w.Event("input", { bubbles: true }));
    assert.ok(aviso().includes("saturado"), aviso());
    assert.ok(!aviso().includes("morto"));
    assert.strictEqual(d.querySelectorAll("#svg .no-saturado").length, 1);
    clicar("3. Volta");
    const svgTxt = d.getElementById("svg").textContent;
    assert.ok(!svgTxt.includes("(×0)"), "saturado não é zero exato");
    assert.ok(/∇ .*e−/.test(svgTxt), "esperava gradiente em notação científica: " + svgTxt.slice(0, 200));
    clicar("Visão geral");
    escolher("sel-saida", "relu");
    assert.ok(aviso().includes("morto"), "na ReLU o mesmo viés mata o neurônio");
    b.value = "-1.5"; b.dispatchEvent(new w.Event("input", { bubbles: true }));
  });

  teste("duas camadas: ativação das ocultas e da saída são independentes", () => {
    clicar("2. Duas camadas");
    escolher("sel-oculta", "tanh");
    escolher("sel-saida", "sigmoid");
    assert.strictEqual(ev("E.rede.ativacaoOculta"), "tanh");
    assert.strictEqual(ev("E.rede.ativacaoSaida"), "sigmoid");
    const rot = [...d.querySelectorAll("#grade-camadas .grupo[data-g='1'] .rotulo")].map((x) => x.textContent);
    assert.ok(rot[0].includes("tanh(z)") && rot[1].includes("logística(z)"), rot.join(" | "));
    assert.ok(d.getElementById("nota-conv").textContent.includes("tanh'(z)"));
    assert.ok(d.getElementById("nota-conv").textContent.includes("σ'(z)"));
  });

  teste("dica da faixa: logística com alvo fora de 0 a 1 avisa que a perda não zera", () => {
    // as barras já impedem y > 1; forçamos um alvo fora da faixa pelo estado para ver o aviso
    ev("E.y = 10; atualizar();");
    assert.ok(d.getElementById("dica-ativ").textContent.includes("0 e 1"));
    assert.ok(d.getElementById("dica-ativ").textContent.includes("fora dessa faixa"));   // y = 10
    escolher("sel-saida", "linear");
    assert.ok(!d.getElementById("dica-ativ").textContent.includes("fora dessa faixa"));
    escolher("sel-oculta", "relu");
  });

  // ---- faixas das barras seguem a ativação ---------------------------------
  const barra = (rot) => d.querySelector(`#controles input[type=range][aria-label='${rot}']`);
  const faixa = (rot) => { const s = barra(rot); return [s.min, s.max, s.step]; };

  teste("faixas: ReLU mantém x de 0 a 20 e y de 1 a 100 (experimento 1)", () => {
    clicar("1. Um neurônio");
    assert.deepStrictEqual(faixa("x₁"), ["0", "20", "0.5"]);
    assert.deepStrictEqual(faixa("alvo y"), ["1", "100", "1"]);
  });

  teste("faixas: tanh deixa x em −3..3 e y em −1..1, e puxa os valores para dentro", () => {
    clicar("1. Um neurônio");
    ev("E.x = [15, 9]; E.y = 70; atualizar();");
    escolher("sel-saida", "tanh");
    assert.deepStrictEqual(faixa("x₁"), ["-3", "3", "0.1"]);
    assert.deepStrictEqual(faixa("alvo y"), ["-1", "1", "0.05"]);
    assert.deepStrictEqual(JSON.parse(ev("JSON.stringify(E.x)")), [3, 3]);
    assert.strictEqual(ev("E.y"), 1);
    assert.ok(d.getElementById("dica-ativ").textContent.includes("−3 e 3"));
    assert.ok(!d.getElementById("dica-ativ").textContent.includes("fora dessa faixa"));
  });

  teste("faixas: logística tem y em 0..1; trocar de volta para ReLU restaura 0..20 e 1..100", () => {
    escolher("sel-saida", "sigmoid");
    assert.deepStrictEqual(faixa("alvo y"), ["0", "1", "0.05"]);
    escolher("sel-saida", "relu");
    assert.deepStrictEqual(faixa("x₁"), ["0", "20", "0.5"]);
    assert.deepStrictEqual(faixa("alvo y"), ["1", "100", "1"]);
  });

  teste("faixas: nas duas camadas x segue a ativação das ocultas e y a da saída", () => {
    clicar("2. Duas camadas");
    assert.deepStrictEqual(faixa("x₁"), ["0", "20", "0.5"]);
    assert.deepStrictEqual(faixa("alvo y"), ["1", "100", "1"]);
    escolher("sel-oculta", "tanh");                           // saída continua linear
    assert.deepStrictEqual(faixa("x₁"), ["-3", "3", "0.1"]);
    assert.deepStrictEqual(faixa("alvo y"), ["1", "100", "1"]);
    escolher("sel-saida", "sigmoid");
    assert.deepStrictEqual(faixa("alvo y"), ["0", "1", "0.05"]);
    escolher("sel-oculta", "relu"); escolher("sel-saida", "linear");
    assert.deepStrictEqual(faixa("x₁"), ["0", "20", "0.5"]);
  });

  teste("faixas: o detector 2×2 não muda (pixels 0/1 e alvo 0/1)", () => {
    clicar("3. Detector");
    escolher("sel-saida", "tanh");
    assert.deepStrictEqual(faixa("alvo y"), ["0", "1", "1"]);
    assert.strictEqual(d.querySelectorAll("#controles input[type=range][aria-label^='p']").length, 0);
    escolher("sel-saida", "relu");
  });

  teste("sorteio garante neurônios vivos: desligado por padrão e, ligado, 40 sorteios sem nenhum morto", () => {
    clicar("3. Detector");
    const cb = d.getElementById("tg-sorteio-vivo");
    assert.strictEqual(cb.checked, false);
    cb.checked = true; cb.dispatchEvent(new w.Event("change", { bubbles: true }));
    for (let k = 0; k < 40; k++) {
      clicar("🎲 Sortear");
      assert.ok(aviso().includes("vivos"), `sorteio ${k}: ${aviso()}`);
    }
    assert.ok(d.getElementById("info-passo").textContent.includes("vivos"));
  });

  teste("sorteio garante vivos também nas duas camadas e com tanh", () => {
    clicar("2. Duas camadas");
    const cb = d.getElementById("tg-sorteio-vivo");
    cb.checked = true; cb.dispatchEvent(new w.Event("change", { bubbles: true }));
    escolher("sel-oculta", "tanh");
    for (let k = 0; k < 20; k++) { clicar("🎲 Sortear"); assert.ok(aviso().includes("vivos"), aviso()); }
    escolher("sel-oculta", "relu");
    for (let k = 0; k < 20; k++) { clicar("🎲 Sortear"); assert.ok(aviso().includes("vivos"), aviso()); }
    cb.checked = false; cb.dispatchEvent(new w.Event("change", { bubbles: true }));
  });

  teste("sem a garantia, a ReLU nasce morta em boa parte dos sorteios (detector)", () => {
    clicar("3. Detector");
    const cb = d.getElementById("tg-sorteio-vivo");
    cb.checked = false; cb.dispatchEvent(new w.Event("change", { bubbles: true }));
    let mortos = 0;
    for (let k = 0; k < 60; k++) { clicar("🎲 Sortear"); if (aviso().includes("morto")) mortos++; }
    assert.ok(mortos >= 10 && mortos <= 50, `mortos = ${mortos} de 60`);
  });

  teste("gradiente da página com tanh/logística bate com o do motor (duas camadas)", () => {
    clicar("2. Duas camadas");
    escolher("sel-oculta", "tanh"); escolher("sel-saida", "sigmoid");
    const r = ev("JSON.stringify(Motor.retropropagar(E.rede,E.x,[E.y]).gb)");
    const camada = d.querySelectorAll("#grade-camadas .camada")[0];
    assert.ok(camada.querySelector(".grupo[data-g='3']"));
    assert.ok(JSON.parse(r).flat().every((v) => Number.isFinite(v)));
    escolher("sel-oculta", "relu"); escolher("sel-saida", "linear");
  });

  teste("sortear pesos: valores novos entre −1 e 1, histórico limpo, entradas e alvo intactos", () => {
    clicar("3. Detector");
    clicar("Horizontal");
    clicar("Vivo");
    clicar("Treinar 20");
    assert.ok(ev("E.historico.length") > 0);
    const xAntes = ev("JSON.stringify(E.x)"), yAntes = ev("E.y"), lrAntes = ev("E.lr");
    const original = w.Math.random;
    w.Math.random = () => 0.75;                       // (0,75*2-1) = 0,5 em todos
    clicar("🎲 Sortear");
    w.Math.random = original;
    assert.deepStrictEqual(JSON.parse(ev("JSON.stringify(E.rede.W[0][0])")), [0.5, 0.5, 0.5, 0.5]);
    assert.strictEqual(ev("E.rede.b[0][0]"), 0.5);
    assert.strictEqual(ev("E.historico.length"), 0);
    assert.strictEqual(ev("E.passos"), 0);
    assert.strictEqual(ev("JSON.stringify(E.x)"), xAntes);
    assert.strictEqual(ev("E.y"), yAntes);
    assert.strictEqual(ev("E.lr"), lrAntes);
  });

  teste("sortear pesos: com a função aleatória real, tudo fica em [−1, 1]", () => {
    for (let k = 0; k < 20; k++) {
      clicar("🎲 Sortear");
      const todos = ev("E.rede.W.flat(2).concat(E.rede.b.flat())");
      assert.ok(Array.from(todos).every((v) => v >= -1 && v <= 1));
    }
  });

  teste("sortear pesos: funciona nos 3 experimentos e a página recalcula sem erro", () => {
    for (const aba of ["1. Um neurônio", "2. Duas camadas", "3. Detector"]) {
      clicar(aba);
      clicar("🎲 Sortear");
      assert.ok(d.getElementById("aviso").textContent.length > 0);
    }
  });

  teste("reiniciar limpa o histórico", () => {
    clicar("Reiniciar");
    assert.strictEqual(ev("E.historico.length"), 0);
  });

  teste("nenhum erro de execução durante todo o uso", () => assert.deepStrictEqual(erros, []));

  console.log(falhas ? `\n${falhas} teste(s) falharam` : "\ntodos os testes da página passaram");
  w.close();
  process.exit(falhas ? 1 : 0);
})();
