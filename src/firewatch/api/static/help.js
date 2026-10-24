"use strict";

const sections = [...document.querySelectorAll("#help-body section")];
const links = [...document.querySelectorAll("#toc a")];

function mark(id) {
  for (const a of links) a.classList.toggle("on", a.getAttribute("href") === `#${id}`);
}

if ("IntersectionObserver" in window) {
  const seen = new Map();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) seen.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0);
    const best = sections.find((s) => (seen.get(s.id) || 0) > 0);
    if (best) mark(best.id);
  }, { rootMargin: "-10% 0px -60% 0px", threshold: [0, 0.01, 0.5] });
  sections.forEach((s) => io.observe(s));
}
if (location.hash) mark(location.hash.slice(1));
else if (sections[0]) mark(sections[0].id);

document.getElementById("help-search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  let shown = 0;
  for (const s of sections) {
    const hit = !q || s.textContent.toLowerCase().includes(q);
    s.hidden = !hit;
    if (hit) shown++;
    if (hit && q) s.querySelectorAll("details").forEach((d) => { d.open = d.textContent.toLowerCase().includes(q); });
  }
  for (const a of links) a.parentElement.hidden = Boolean(document.getElementById(a.getAttribute("href").slice(1)).hidden);
  document.getElementById("no-match").hidden = shown > 0;
});

for (const pre of document.querySelectorAll("pre")) {
  const button = document.createElement("button");
  button.className = "copy";
  button.type = "button";
  button.textContent = "Copy";
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(pre.querySelector("code").textContent);
      button.textContent = "Copied";
    } catch (err) {
      const range = document.createRange();
      range.selectNodeContents(pre.querySelector("code"));
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      button.textContent = "Selected";
    }
    setTimeout(() => { button.textContent = "Copy"; }, 1600);
  });
  pre.prepend(button);
}
