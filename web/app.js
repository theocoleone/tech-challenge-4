"use strict";

const SITUACOES = {
  ok: "com evidência",
  sem_evidencia: "sem evidência",
  evidencia_fraca: "evidência insuficiente",
  filtro_sem_resultado: "recorte vazio",
  fora_escopo: "fora do escopo",
};

const CAMPOS = ["review_score_min", "review_score_max", "categoria_pt", "ano_mes_compra_min",
  "ano_mes_compra_max", "order_status", "entregue_com_atraso", "uf_cliente", "valor_min", "valor_max"];
const NUMERICOS = new Set(["review_score_min", "review_score_max", "ano_mes_compra_min",
  "ano_mes_compra_max", "valor_min", "valor_max"]);

const $ = (id) => document.getElementById(id);
const texto = (v) => String(v ?? "");

function filtrosDoFormulario() {
  const filtros = {};
  for (const campo of CAMPOS) {
    const valor = $(campo).value.trim();
    if (!valor) continue;
    if (NUMERICOS.has(campo)) filtros[campo] = Number(valor);
    else if (valor === "true" || valor === "false") filtros[campo] = valor === "true";
    else filtros[campo] = campo === "uf_cliente" ? valor.toUpperCase() : valor;
  }
  return filtros;
}

function mes(aaaamm) {
  const s = texto(aaaamm);
  return s.length === 6 ? `${s.slice(4)}/${s.slice(0, 4)}` : s;
}

function cartaoEvidencia(e) {
  const entrega = !e.entregue
    ? '<span class="atrasado">pedido não entregue</span>'
    : (e.entregue_com_atraso ? '<span class="atrasado">entrega com atraso</span>' : "entrega no prazo");
  return `<article class="cartao" id="ev-${e.review_id}">
    <blockquote>${escapar(e.texto)}</blockquote>
    <div class="cartao-meta">
      <span class="nota" data-n="${e.nota}">nota ${e.nota}</span>
      <span>${escapar(e.categoria)}</span>
      <span>compra ${mes(e.mes_compra)}</span>
      <span>${entrega}</span>
      ${e.score_rerank === null || e.score_rerank === undefined ? "" :
        `<span title="relevância dada pelo re-ranker">rerank ${e.score_rerank}</span>`}
    </div>
  </article>`;
}

function escapar(s) {
  const div = document.createElement("div");
  div.textContent = texto(s);
  return div.innerHTML;
}

function mostrarContagem(c) {
  const bloco = $("contagem");
  if (!c) { bloco.hidden = true; return; }
  const rotulos = {
    nota_alta: "avaliações com nota 4 ou 5",
    nota_baixa: "avaliações com nota 1 ou 2",
    todas: "avaliações com nota 1, 2, 4 ou 5",
  };
  const rotulo = rotulos[c.polaridade] || "avaliações com tema";
  $("base-contagem").textContent =
    `${c.total_com_tema.toLocaleString("pt-BR")} ${rotulo} no recorte (de ${c.total_no_recorte.toLocaleString("pt-BR")} com comentário). ` +
    `As contagens usam todos os comentários classificados em temas no recorte, não só as avaliações mostradas ao lado.`;
  const maior = Math.max(...c.temas.map((t) => t.pct), 1);
  const barras = $("barras");
  barras.dataset.pol = c.polaridade;
  barras.innerHTML = c.temas.map((t) => `<li>
      <span>${escapar(t.tema)} <small style="color:var(--tinta-fraca)">${escapar(t.macro)}</small></span>
      <span class="trilho"><span class="preenchida" style="width:${(t.pct / maior) * 100}%"></span></span>
      <span class="valor">${t.n.toLocaleString("pt-BR")} · ${String(t.pct).replace(".", ",")}%</span>
    </li>`).join("");
  bloco.hidden = false;
}

function mostrar(r) {
  $("situacao").textContent = SITUACOES[r.status] || r.status;
  $("situacao").dataset.s = r.status;
  const tempos = Object.entries(r.tempos || {}).map(([k, v]) => `${k} ${v}s`).join(" · ");
  $("tempo").textContent = tempos;
  $("texto-resposta").textContent = r.resposta;

  const aplicado = Object.entries(r.filtros_aplicados || {});
  $("aplicado").innerHTML = [
    `<li>busca: “${escapar(r.pergunta_reescrita)}”</li>`,
    `<li>tipo: ${r.tipo === "agregada" ? "frequência" : "específica"}</li>`,
    ...aplicado.map(([k, v]) => `<li>${escapar(k)} = ${escapar(v)}</li>`),
    `<li>${(r.n_no_recorte ?? 0).toLocaleString("pt-BR")} avaliações no recorte</li>`,
  ].join("");

  $("insights").innerHTML = (r.insights || []).map((i) => {
    const chips = i.review_ids.map((id) => `<a href="#ev-${id}" title="ver a avaliação">${id.slice(0, 6)}</a>`).join("");
    return `<li>${escapar(i.afirmacao)}<span class="citacoes">${chips}</span></li>`;
  }).join("");

  const evidencias = r.evidencias || [];
  const notasRecusa = {
    sem_evidencia: "As avaliações abaixo foram as mais próximas, mas nenhuma passou pelo portão de relevância.",
    evidencia_fraca: "Há alguma evidência, mas ela não é suficiente para sustentar uma conclusão geral.",
    filtro_sem_resultado: "O recorte solicitado não contém avaliações com comentário.",
    fora_escopo: "A pergunta não pertence ao domínio coberto pela base.",
  };
  const recusou = r.status !== "ok";
  $("titulo-evidencias").textContent = recusou ? "Avaliações inspecionadas" : "Avaliações usadas como evidência";
  $("nota-evidencias").textContent = notasRecusa[r.status] || "";
  $("nota-evidencias").hidden = !notasRecusa[r.status];
  $("evidencias").innerHTML = evidencias.map(cartaoEvidencia).join("");
  $("bloco-evidencias").hidden = evidencias.length === 0;
  mostrarContagem(r.contagem);
  $("saida").hidden = false;
}

async function perguntar(evento) {
  evento.preventDefault();
  const pergunta = $("pergunta").value.trim();
  if (!pergunta) return;
  const botao = $("enviar");
  botao.disabled = true;
  botao.textContent = "consultando…";
  $("erro").hidden = true;
  try {
    const filtros = filtrosDoFormulario();
    const resposta = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pergunta, filtros: Object.keys(filtros).length ? filtros : null }),
    });
    if (!resposta.ok) throw new Error(`a API respondeu ${resposta.status}`);
    mostrar(await resposta.json());
  } catch (erro) {
    $("erro").textContent = `Não foi possível responder: ${erro.message}. Confira se a API está no ar (make api).`;
    $("erro").hidden = false;
  } finally {
    botao.disabled = false;
    botao.textContent = "Perguntar";
  }
}

async function iniciar() {
  $("forma").addEventListener("submit", perguntar);
  $("exemplos").addEventListener("click", (e) => {
    if (!e.target.dataset.q) return;
    $("pergunta").value = e.target.dataset.q;
    $("forma").requestSubmit();
  });
  $("limpar").addEventListener("click", () => CAMPOS.forEach((c) => ($(c).value = "")));
  $("pergunta").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("forma").requestSubmit(); }
  });

  try {
    const [saude, categorias] = await Promise.all([
      fetch("/health").then((r) => r.json()),
      fetch("/categorias").then((r) => r.json()),
    ]);
    $("n-documentos").textContent = saude.documentos.toLocaleString("pt-BR");
    $("estado").innerHTML = [
      ["avaliações", saude.documentos.toLocaleString("pt-BR")],
      ["temas", saude.temas],
      ["busca", saude.modelos.embedding.split(".").pop()],
      ["resposta", saude.modelos.geracao.split(".").pop()],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${escapar(v)}</dd>`).join("");
    $("categoria_pt").insertAdjacentHTML("beforeend",
      categorias.map((c) => `<option value="${escapar(c)}">${escapar(c.replace(/_/g, " "))}</option>`).join(""));
  } catch {
    $("erro").textContent = "A API não respondeu. Rode make api e recarregue a página.";
    $("erro").hidden = false;
  }
}

iniciar();
