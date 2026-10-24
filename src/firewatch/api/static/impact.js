"use strict";

// Homepage opening: photo slideshow, floating nav and scroll reveals.
(function () {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SLIDE_MS = 6000;

  const nav = document.querySelector(".nav");
  const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 40);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const slides = [...document.querySelectorAll(".ih-slide")];
  const dots = document.getElementById("ih-dots");
  const cap = document.getElementById("ih-cap");
  const credit = document.getElementById("ih-credit");
  let current = -1;
  let timer = null;

  dots.style.setProperty("--slide-ms", SLIDE_MS + "ms");
  slides.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", "Show photo " + (i + 1) + " of " + slides.length);
    b.addEventListener("click", () => { show(i); restart(); });
    dots.appendChild(b);
  });
  const dotButtons = [...dots.children];

  function show(i) {
    if (i === current) return;
    if (current >= 0) {
      slides[current].classList.remove("on");
      dotButtons[current].classList.remove("on");
    }
    current = i;
    const s = slides[i];
    s.classList.add("on");
    // Restart the progress bar on the active dot.
    const d = dotButtons[i];
    d.classList.remove("on");
    void d.offsetWidth;
    d.classList.add("on");
    d.setAttribute("aria-current", "true");
    dotButtons.forEach((o) => { if (o !== d) o.removeAttribute("aria-current"); });
    cap.textContent = s.dataset.caption || "";
    credit.innerHTML = "";
    const a = document.createElement("a");
    a.href = s.dataset.source;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = "Photo: " + s.dataset.credit;
    credit.appendChild(a);
    // Warm the next photo so the crossfade never shows a blank frame.
    const next = slides[(i + 1) % slides.length].querySelector("img");
    if (next.loading === "lazy") next.loading = "eager";
  }

  function restart() {
    clearInterval(timer);
    if (reduced) { dots.classList.add("paused"); return; }
    timer = setInterval(() => show((current + 1) % slides.length), SLIDE_MS);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearInterval(timer);
    else restart();
  });

  // A photo that fails to load (the host moved it) drops out of the rotation.
  slides.forEach((s) => s.querySelector("img").addEventListener("error", () => {
    const i = slides.indexOf(s);
    if (i < 0 || slides.length < 2) return;
    s.remove();
    dotButtons[i].remove();
    slides.splice(i, 1);
    dotButtons.splice(i, 1);
    if (current >= i) current = Math.max(-1, current - 1);
    if (current < 0) show(0);
  }));
  document.querySelectorAll(".reel img").forEach((img) => img.addEventListener("error", () => img.closest("figure").remove()));

  show(0);
  restart();

  // Reveal blocks as they scroll into view.
  const targets = document.querySelectorAll(".section-head, .mission-text, .stats li, .now .wrap > *, .paths .path, .how-steps li, .closer h2, .closer .ih-actions");
  if (reduced || !("IntersectionObserver" in window)) return;
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      io.unobserve(e.target);
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
  targets.forEach((t, i) => {
    t.classList.add("reveal");
    const siblings = t.parentElement ? [...t.parentElement.children].filter((c) => c.matches(t.tagName)) : [];
    const n = siblings.indexOf(t);
    if (n > 0) t.style.transitionDelay = Math.min(n, 4) * 0.08 + "s";
    io.observe(t);
  });
})();
