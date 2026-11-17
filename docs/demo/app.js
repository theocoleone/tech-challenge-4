"use strict";

const SITUACOES = {
  ok: "com evidência",
  sem_evidencia: "sem evidência",
  evidencia_fraca: "evidência insuficiente",
  filtro_sem_resultado: "recorte vazio",
  fora_escopo: "fora do escopo",
};

const $ = (id) => document.getElementById(id);
let dados = null;

function escapar(s) {
  const d = document.createElement("div");
  d.textContent = String(s ?? "");
  return d.innerHTML;
}

const mes = (v) => {
  const s = String(v ?? "");
  return s.length === 6 ? `${s.slice(4)}/${s.slice(0, 4)}` : s;
};

function cartaoEvidencia(e) {
  const entrega = e.entregue === false
    ? "não entregue"
    : e.entregue_com_atraso ? '<span class="atrasado">entrega com atraso</span>' : "entrega no prazo";
  const score = e.score_rerank == null ? "" :
    `<span title="relevância dada pelo re-ranqueador">rerank ${e.score_rerank}</span>`;
  return `<article class="cartao" id="ev-${e.review_id}">
    <blockquote>${escapar(e.texto)}</blockquote>
    <div class="cartao-meta">
      <span class="nota" data-n="${e.nota}">nota ${e.nota}</span>
      <span>${escapar(e.categoria)}</span>
      <span>compra ${mes(e.mes_compra)}</span>
      <span>${entrega}</span>
      ${score}
    </div>
  </article>`;
}

function mostrarContagem(c) {
  const bloco = $("contagem");
  if (!c || !c.temas?.length) { bloco.hidden = true; return; }
  const rotulo = { nota_baixa: "avaliações com nota 1 ou 2", nota_alta: "avaliações com nota 4 ou 5" }[c.polaridade]
    || "avaliações com tema";
  $("base-contagem").textContent =
    `${c.total_com_tema.toLocaleString("pt-BR")} ${rotulo} no recorte (de ${c.total_no_recorte.toLocaleString("pt-BR")} com comentário). ` +
    "As contagens vêm da base inteira, não das avaliações mostradas ao lado.";
  const maior = Math.max(...c.temas.map((t) => t.pct), 1);
  const barras = $("barras");
  barras.dataset.pol = c.polaridade;
  barras.innerHTML = c.temas.map((t) => `<li>
      <span>${escapar(t.tema)} <small>${escapar(t.macro)}</small></span>
      <span class="trilho"><span class="preenchida" style="width:${(t.pct / maior) * 100}%"></span></span>
      <span class="valor">${t.n.toLocaleString("pt-BR")} · ${String(t.pct).replace(".", ",")}%</span>
    </li>`).join("");
  bloco.hidden = false;
}

function mostrar(p) {
  $("situacao").textContent = SITUACOES[p.status] || p.status;
  $("situacao").dataset.s = p.status;
  $("tempo").textContent = Object.entries(p.tempos || {}).map(([k, v]) => `${k} ${v}s`).join(" · ");
  $("pergunta-atual").textContent = p.pergunta;
  $("texto-resposta").textContent = p.resposta;

  $("aplicado").innerHTML = [
    `<li>busca: “${escapar(p.pergunta_reescrita)}”</li>`,
    `<li>tipo: ${p.tipo === "agregada" ? "frequência" : "específica"}</li>`,
    ...Object.entries(p.filtros_aplicados || {}).map(([k, v]) => `<li>${escapar(k)} = ${escapar(v)}</li>`),
    `<li>${(p.n_no_recorte ?? 0).toLocaleString("pt-BR")} avaliações no recorte</li>`,
  ].join("");

  $("insights").innerHTML = (p.insights || []).map((i) => {
    const chips = i.review_ids.map((id) => `<a href="#ev-${id}" title="ver a avaliação">${id.slice(0, 6)}</a>`).join("");
    return `<li>${escapar(i.afirmacao)}<span class="citacoes">${chips}</span></li>`;
  }).join("");

  const recusou = p.status !== "ok";
  $("titulo-evidencias").textContent = recusou ? "Avaliações mais próximas, descartadas" : "Avaliações usadas como evidência";
  $("nota-evidencias").textContent = recusou
    ? "Nenhuma passou do limiar de relevância, então a resposta não se apoia nelas. Ficam visíveis para mostrar por que o sistema recusou."
    : "";
  $("nota-evidencias").hidden = !recusou;
  $("evidencias").innerHTML = (p.evidencias || []).map(cartaoEvidencia).join("");
  $("bloco-evidencias").hidden = (p.evidencias || []).length === 0;
  mostrarContagem(p.contagem);

  document.querySelectorAll("#lista button").forEach((b) => b.setAttribute("aria-current", String(b.dataset.id === p.id)));
  $("saida").scrollIntoView({ behavior: "smooth", block: "start" });
}

function montarLista() {
  const grupos = new Map();
  for (const p of dados.perguntas) {
    if (!grupos.has(p.grupo)) grupos.set(p.grupo, []);
    grupos.get(p.grupo).push(p);
  }
  $("lista").innerHTML = [...grupos].map(([grupo, ps]) => `
    <div class="grupo">
      <h3>${escapar(grupo)} <span>${ps.length}</span></h3>
      ${ps.map((p) => `<button type="button" data-id="${p.id}" data-s="${p.status}">
        <span class="ponto" data-s="${p.status}" aria-hidden="true"></span>${escapar(p.pergunta)}
      </button>`).join("")}
    </div>`).join("");

  $("lista").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) mostrar(dados.perguntas.find((p) => p.id === b.dataset.id));
  });
}

async function iniciar() {
  try {
    dados = await (await fetch("respostas.json")).json();
  } catch {
    $("texto-resposta").textContent = "Não foi possível carregar as respostas (respostas.json).";
    return;
  }
  const r = dados.resumo;
  $("gerado").textContent = new Date(r.gerado_em + "T12:00:00").toLocaleDateString("pt-BR");
  $("estado").innerHTML = [
    ["perguntas", r.n_perguntas],
    ["ramos corretos", `${r.ramos_corretos}/${r.n_perguntas}`],
    ["citações válidas", `${Math.round(r.citacoes_validas * 100)}%`],
    ["fidelidade", `${Math.round(r.fidelidade * 100)}%`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${escapar(v)}</dd>`).join("");

  montarLista();
  const inicial = dados.perguntas.find((p) => p.tipo === "agregada") || dados.perguntas[0];
  mostrar(inicial);
}

iniciar();
