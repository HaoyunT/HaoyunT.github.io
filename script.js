document.addEventListener('DOMContentLoaded', function() {
    const mainArea = document.querySelector('.main-content-area');

    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(event) {
            const href = this.getAttribute('href');
            if (href === '#') {
                return;
            }

            const target = document.querySelector(href);
            if (target) {
                event.preventDefault();
                const headerOffset = 80;
                const targetTop = target.getBoundingClientRect().top + window.pageYOffset - headerOffset;
                window.scrollTo({ top: targetTop, behavior: 'smooth' });
            }
        });
    });

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    if (mainArea) {
        const revealTargets = [];
        let currentGroup = null;

        Array.from(mainArea.children).forEach(child => {
            if (child.classList.contains('top-nav') || child.id === 'footer') {
                return;
            }

            if (child.tagName === 'H2') {
                currentGroup = document.createElement('div');
                currentGroup.className = 'scroll-reveal';
                mainArea.insertBefore(currentGroup, child);
                revealTargets.push(currentGroup);
            }

            if (currentGroup) {
                currentGroup.appendChild(child);
            }
        });

        revealTargets.forEach(target => observer.observe(target));
    }

    const navLinks = document.querySelectorAll('.top-nav-links a');
    const sectionHeadings = document.querySelectorAll('.main-content-area h2[id]');

    function updateActiveNav() {
        let current = '';

        sectionHeadings.forEach(heading => {
            if (window.pageYOffset >= heading.offsetTop - 140) {
                current = heading.id;
            }
        });

        navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === `#${current}`);
        });
    }

    window.addEventListener('scroll', updateActiveNav, { passive: true });
    updateActiveNav();

    const emailElement = document.querySelector('a[href="mailto:haoyuntang224@163.com"]');
    if (emailElement) {
        emailElement.addEventListener('click', function(event) {
            event.preventDefault();
            copyText('haoyuntang224@163.com', 'Email address copied to clipboard.');
        });
    }

    const footerText = document.querySelector('#footer-text p');
    if (footerText) {
        const now = new Date();
        const monthNames = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ];
        footerText.textContent = `© ${now.getFullYear()} Haoyun Tang | Last updated: ${monthNames[now.getMonth()]} ${now.getFullYear()}`;
    }
});

function copyText(text, message) {
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => showNotification(message));
        return;
    }

    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    try {
        document.execCommand('copy');
        showNotification(message);
    } finally {
        document.body.removeChild(textArea);
    }
}

function showNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'site-notification';
    notification.textContent = message;
    document.body.appendChild(notification);

    requestAnimationFrame(() => {
        notification.classList.add('visible');
    });

    setTimeout(() => {
        notification.classList.remove('visible');
        setTimeout(() => notification.remove(), 250);
    }, 2200);
}
