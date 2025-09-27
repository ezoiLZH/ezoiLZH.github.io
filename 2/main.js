window.addEventListener('scroll', function () {
    const y = window.scrollY;
    document.querySelector('.parallax-bg').style.backgroundPosition = `center ${-y * 0.3}px`;
});
