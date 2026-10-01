(function () {
  var page = (location.pathname.split("/").pop() || "index.html").toLowerCase();

  var icons = {
    home: '<path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/>',
    dash: '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    fore: '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    about: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>'
  };
  var links = [
    ["index.html", "Home", icons.home],
    ["dashboard.html", "Dashboard", icons.dash],
    ["forecast.html", "Forecast", icons.fore],
    ["about.html", "About", icons.about]
  ];

  var logoSvg = '<svg viewBox="0 0 32 32" fill="none" aria-hidden="true"><rect width="32" height="32" rx="9" fill="#d75b91"/><path d="M5 21l6-7 5 4 6-9 5 5" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/><circle cx="27" cy="13" r="2" fill="#ffc1d9"/></svg>';

  var items = links.map(function (l) {
    var active = page === l[0];
    return '<li><a href="' + l[0] + '"' + (active ? ' class="active" aria-current="page"' : "") + '>' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + l[2] + '</svg>' +
      l[1] + '</a></li>';
  }).join("");

  var html =
    '<div class="eco-top">' +
      '<button class="eco-burger" id="ecoBurger" aria-label="Open navigation menu" aria-expanded="false">☰</button>' +
      '<a class="eco-logo" href="index.html">' + logoSvg + '<div>Eco<span>Forecast</span></div></a>' +
    '</div>' +
    '<div class="eco-scrim" id="ecoScrim"></div>' +
    '<aside class="eco-side" id="ecoSide" aria-label="Main navigation">' +
      '<a class="eco-logo" href="index.html">' + logoSvg + '<div>Eco<span>Forecast</span></div></a>' +
      '<nav class="eco-nav"><div class="eco-label">Menu</div><ul>' + items + '</ul></nav>' +
      '<div class="eco-card">' +
        '<small>Forecast model</small><b>XGBoost</b><em>1 hour ahead · time series</em>' +
        '<svg viewBox="0 0 200 56" fill="none" aria-hidden="true">' +
          '<path d="M2 44 C22 40 34 18 54 26 S88 50 104 32" stroke="#fff" stroke-width="3" stroke-linecap="round"/>' +
          '<path d="M104 32 C124 22 138 6 160 14 S190 8 198 4" stroke="#ffc1d9" stroke-width="3" stroke-linecap="round" stroke-dasharray="3 8"/>' +
          '<circle cx="104" cy="32" r="5" fill="#7d123f" stroke="#fff" stroke-width="2.5"/>' +
        '</svg>' +
      '</div>' +
      '<div class="eco-credit">Energy Consumption Forecasting<br>Developed by Ashi Gupta</div>' +
    '</aside>';

  document.body.insertAdjacentHTML("afterbegin", html);

  var side = document.getElementById("ecoSide"),
      scrim = document.getElementById("ecoScrim"),
      burger = document.getElementById("ecoBurger");

  function setOpen(open) {
    side.classList.toggle("open", open);
    scrim.classList.toggle("show", open);
    burger.setAttribute("aria-expanded", open);
    burger.textContent = open ? "✕" : "☰";
  }
  burger.addEventListener("click", function () { setOpen(!side.classList.contains("open")); });
  scrim.addEventListener("click", function () { setOpen(false); });
  side.querySelectorAll("a").forEach(function (a) { a.addEventListener("click", function () { setOpen(false); }); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") setOpen(false); });
  window.addEventListener("resize", function () { if (window.innerWidth > 900) setOpen(false); });
})();