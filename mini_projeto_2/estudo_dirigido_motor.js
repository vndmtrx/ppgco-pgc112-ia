/* Motor numérico do estudo dirigido (estudo_dirigido.html).
 *
 * Funções puras para uma rede feed-forward pequena. A ativação das camadas ocultas e a da saída
 * podem ser ReLU, tanh ou logística (a saída também pode ser linear). A perda é
 * Loss = ½ · média_das_amostras( Σ (a_saída − y)² ).
 *
 * Convenções (iguais às da METODOLOGIA.md, seções 4.1 e 4.5):
 *   z_l = W_l · a_{l-1} + b_l            a_l = f(z_l)
 *   δ_L = (a_L − y) ⊙ f'(z_L)            δ_l = (W_{l+1}ᵀ · δ_{l+1}) ⊙ f'(z_l)
 *   ∇W_l = δ_l · a_{l-1}ᵀ                ∇b_l = δ_l
 *   ReLU'(z) = 1 se z > 0, senão 0       (com z ≤ 0 o gradiente é exatamente zero: neurônio morto)
 *   tanh'(z) = 1 − tanh²(z)  (máx. 1)    logística'(z) = σ(z)(1 − σ(z))  (máx. 0,25)
 *   Nessas duas a derivada nunca chega a 0, mas encolhe longe de z = 0 (neurônio saturado).
 *
 * Roda no navegador (window.Motor) e no Node (module.exports), para os testes.
 */
(function (raiz) {
  "use strict";

  function relu(z) { return z > 0 ? z : 0; }
  function derivadaRelu(z) { return z > 0 ? 1 : 0; }
  function logistica(z) { return 1 / (1 + Math.exp(-z)); }

  /**
   * Funções de ativação. `dfMax` é o maior valor da derivada (serve para medir a saturação).
   * ReLU: derivada exatamente 0 para z ≤ 0 (neurônio "morto").
   * tanh e logística: derivada nunca chega a 0, mas fica minúscula longe de z = 0 (neurônio "saturado").
   */
  const ATIVACOES = {
    relu:    { nome: "ReLU",      f: relu,      df: derivadaRelu, dfMax: 1 },
    tanh:    { nome: "tanh",      f: Math.tanh, df: (z) => { const t = Math.tanh(z); return 1 - t * t; }, dfMax: 1 },
    sigmoid: { nome: "logística", f: logistica, df: (z) => { const s = logistica(z); return s * (1 - s); }, dfMax: 0.25 },
    linear:  { nome: "linear",    f: (z) => z,  df: () => 1, dfMax: 1 },
  };
  // Um neurônio não-ReLU conta como saturado quando f'(z) cai abaixo de 10% do máximo.
  const LIMIAR_SATURACAO = 0.1;

  function neuronioVivo(modo, z) {
    if (modo === "relu") return z > 0;
    const a = ATIVACOES[modo];
    return a.df(z) >= LIMIAR_SATURACAO * a.dfMax;
  }

  /**
   * Cria a rede. `camadas` = [entradas, ocultas..., saidas]. `W[l][i][j]`: neurônio i da camada l, entrada j.
   * `ativacaoSaida` e `ativacaoOculta`: "relu" | "tanh" | "sigmoid" | "linear" (a oculta não pode ser "linear").
   */
  function criarRede(camadas, ativacaoSaida, W, b, ativacaoOculta) {
    const rede = {
      camadas: camadas.slice(),
      ativacaoSaida: ATIVACOES[ativacaoSaida] ? ativacaoSaida : "relu",
      ativacaoOculta: ATIVACOES[ativacaoOculta] && ativacaoOculta !== "linear" ? ativacaoOculta : "relu",
      W: [],
      b: [],
    };
    for (let l = 1; l < camadas.length; l++) {
      const m = camadas[l], n = camadas[l - 1];
      rede.W.push(W ? W[l - 1].map((linha) => linha.slice()) : Array.from({ length: m }, () => new Array(n).fill(0)));
      rede.b.push(b ? b[l - 1].slice() : new Array(m).fill(0));
    }
    return rede;
  }

  function clonar(rede) {
    return criarRede(rede.camadas, rede.ativacaoSaida, rede.W, rede.b, rede.ativacaoOculta);
  }

  /** Nome da ativação da camada `l` (índice em W: 0 = primeira camada oculta; a última é a saída). */
  function ativacaoDaCamada(rede, l) {
    return l === rede.W.length - 1 ? rede.ativacaoSaida : rede.ativacaoOculta;
  }

  /**
   * Passagem para frente. Devolve z[l], a[l] (a[0] = x), f'(z) em `deriv[l]` e `viva[l][i]`
   * (ReLU: z > 0; tanh e logística: f'(z) ainda ≥ 10% do máximo; linear: sempre viva).
   */
  function avancar(rede, x) {
    const a = [x.slice()];
    const z = [];
    const viva = [];
    const deriv = [];
    for (let l = 0; l < rede.W.length; l++) {
      const zl = rede.W[l].map((linha, i) => linha.reduce((s, w, j) => s + w * a[l][j], rede.b[l][i]));
      const modo = ativacaoDaCamada(rede, l);
      const ativ = ATIVACOES[modo];
      z.push(zl);
      viva.push(zl.map((v) => neuronioVivo(modo, v)));
      deriv.push(zl.map((v) => ativ.df(v)));
      a.push(zl.map((v) => ativ.f(v)));
    }
    return { z, a, viva, deriv, saida: a[a.length - 1] };
  }

  /** Retropropagação de UMA amostra. Devolve δ, ∇W, ∇b e a perda ½Σ(a−y)². */
  function retropropagar(rede, x, y) {
    const f = avancar(rede, x);
    const L = rede.W.length;
    const delta = new Array(L);
    const saida = f.a[L];
    let perda = 0;
    delta[L - 1] = saida.map((av, i) => {
      const erro = av - y[i];
      perda += 0.5 * erro * erro;
      return erro * f.deriv[L - 1][i];
    });
    for (let l = L - 2; l >= 0; l--) {
      delta[l] = f.z[l].map((_, i) => {
        let soma = 0;
        for (let k = 0; k < rede.W[l + 1].length; k++) soma += rede.W[l + 1][k][i] * delta[l + 1][k];
        return soma * f.deriv[l][i];
      });
    }
    // `limpa` troca -0 por 0: sem isso a página mostraria "-0.00" nos gradientes bloqueados.
    const limpa = (v) => (v === 0 ? 0 : v);
    const gW = rede.W.map((Wl, l) => Wl.map((linha, i) => linha.map((_, j) => limpa(delta[l][i] * f.a[l][j]))));
    const gb = delta.map((d) => d.map(limpa));
    return { frente: f, delta: delta.map((d) => d.map(limpa)), gW, gb, perda };
  }

  /** Média dos gradientes e da perda sobre várias amostras [{x, y}]. */
  function gradienteDoLote(rede, amostras) {
    let acum = null;
    let perda = 0;
    for (const { x, y } of amostras) {
      const r = retropropagar(rede, x, y);
      perda += r.perda;
      if (!acum) {
        acum = { gW: r.gW.map((M) => M.map((l) => l.slice())), gb: r.gb.map((v) => v.slice()) };
      } else {
        r.gW.forEach((M, l) => M.forEach((linha, i) => linha.forEach((v, j) => { acum.gW[l][i][j] += v; })));
        r.gb.forEach((v, l) => v.forEach((g, i) => { acum.gb[l][i] += g; }));
      }
    }
    const n = amostras.length;
    acum.gW.forEach((M) => M.forEach((linha) => linha.forEach((_, j, arr) => { arr[j] /= n; })));
    acum.gb.forEach((v) => v.forEach((_, i, arr) => { arr[i] /= n; }));
    return { gW: acum.gW, gb: acum.gb, perda: perda / n };
  }

  /** Um passo de SGD (muda a rede). Devolve a perda ANTES do passo e se algum peso mudou. */
  function aplicarPasso(rede, amostras, taxa) {
    const g = gradienteDoLote(rede, amostras);
    let mudou = false;
    g.gW.forEach((M, l) => M.forEach((linha, i) => linha.forEach((v, j) => {
      if (v !== 0) mudou = true;
      rede.W[l][i][j] -= taxa * v;
    })));
    g.gb.forEach((vet, l) => vet.forEach((v, i) => {
      if (v !== 0) mudou = true;
      rede.b[l][i] -= taxa * v;
    }));
    return { perda: g.perda, mudou, gW: g.gW, gb: g.gb };
  }

  /** Perda média ½Σ(a−y)² sem mexer na rede. */
  function perdaDoLote(rede, amostras) {
    let s = 0;
    for (const { x, y } of amostras) {
      const a = avancar(rede, x).saida;
      a.forEach((av, i) => { s += 0.5 * (av - y[i]) * (av - y[i]); });
    }
    return s / amostras.length;
  }

  const Motor = { relu, derivadaRelu, logistica, ATIVACOES, LIMIAR_SATURACAO, ativacaoDaCamada, criarRede, clonar, avancar, retropropagar, gradienteDoLote, aplicarPasso, perdaDoLote };

  if (typeof module !== "undefined" && module.exports) module.exports = Motor;
  else raiz.Motor = Motor;
})(typeof window !== "undefined" ? window : this);
