const card = document.getElementById("card");
const fills = document.querySelectorAll(".fill");
const typeText = document.querySelector(".type");

// 🎯 Animate stat bars
fills.forEach((fill) => {
    let value = parseInt(fill.getAttribute("data-value")) || 0;

    // normalize (max 150 for nice UI)
    let width = Math.min(value, 150) / 150 * 100;

    setTimeout(() => {
        fill.style.width = width + "%";
    }, 300);
});

// 🎯 Apply Pokémon type color
if (typeText) {
    let type = typeText.innerText.toLowerCase();
    card.classList.add(type);
}

// 🎯 3D + dynamic lighting
card.addEventListener("mousemove", (e) => {
    const rect = card.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const rotateX = (y - centerY) / 12;
    const rotateY = (centerX - x) / 12;

    card.style.transform = `
        rotateX(${rotateX}deg)
        rotateY(${rotateY}deg)
        scale(1.08)
    `;

    // 🌟 moving light
    const glowX = (x / rect.width) * 100;
    const glowY = (y / rect.height) * 100;

    card.style.backgroundImage = `
        radial-gradient(circle at ${glowX}% ${glowY}%,
        rgba(255,255,255,0.7),
        transparent 40%)
    `;
});

card.addEventListener("mousemove", (e) => {
    const rect = card.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const rotateX = (y - centerY) / 12;
    const rotateY = (centerX - x) / 12;

    card.style.transform = `
        rotateX(${rotateX}deg)
        rotateY(${rotateY}deg)
        scale(1.08)
    `;

    // 🌈 rainbow glow following mouse
    const glowX = (x / rect.width) * 100;
    const glowY = (y / rect.height) * 100;

    card.style.boxShadow = `
        0 0 30px rgba(255,0,150,0.6),
        0 0 60px rgba(0,255,255,0.6),
        0 0 90px rgba(255,255,0,0.6)
    `;

    card.style.background = `
        radial-gradient(circle at ${glowX}% ${glowY}%,
        rgba(255,255,255,0.2),
        #000 60%)
    `;
});