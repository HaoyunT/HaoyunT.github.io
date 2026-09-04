document.addEventListener('DOMContentLoaded', function() {
    const mainArea = document.querySelector('.main-content-area');
    const scrollProgress = document.querySelector('.scroll-progress');

    function updateScrollProgress() {
        if (!scrollProgress) {
            return;
        }

        const scrollRange = document.documentElement.scrollHeight - window.innerHeight;
        const progress = scrollRange > 0 ? Math.min(window.scrollY / scrollRange, 1) : 0;
        scrollProgress.style.transform = `scaleX(${progress})`;
    }

    window.addEventListener('scroll', updateScrollProgress, { passive: true });
    window.addEventListener('resize', updateScrollProgress);
    updateScrollProgress();

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
                observer.unobserve(entry.target);
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

        const itemSelectors = [
            '.news-list li',
            '.publication-entry',
            '.xp-item',
            '.awards-item',
            '.project-item'
        ];

        const itemTargets = Array.from(mainArea.querySelectorAll(itemSelectors.join(',')));

        itemTargets.forEach((item, index) => {
            item.classList.add('reveal-item');
            item.style.setProperty('--reveal-delay', `${(index % 5) * 65}ms`);
        });

        const allRevealTargets = [...revealTargets, ...itemTargets];
        allRevealTargets.forEach(target => observer.observe(target));

        function revealVisibleTargets() {
            const revealLine = window.innerHeight * 0.92;

            allRevealTargets.forEach(target => {
                if (target.classList.contains('visible')) {
                    return;
                }

                const rect = target.getBoundingClientRect();
                if (rect.top <= revealLine && rect.bottom >= 0) {
                    target.classList.add('visible');
                    observer.unobserve(target);
                }
            });
        }

        window.addEventListener('scroll', revealVisibleTargets, { passive: true });
        window.addEventListener('resize', revealVisibleTargets);
        revealVisibleTargets();
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

    document.querySelectorAll('.bib-toggle').forEach(button => {
        const panel = document.getElementById(button.getAttribute('aria-controls'));
        if (!panel) {
            return;
        }

        button.addEventListener('click', function() {
            const willOpen = panel.hidden;
            panel.hidden = !willOpen;
            button.setAttribute('aria-expanded', String(willOpen));
        });
    });

    document.querySelectorAll('.copy-bib').forEach(button => {
        button.addEventListener('click', function() {
            const code = button.parentElement.querySelector('code');
            if (code) {
                copyText(code.textContent.trim(), 'BibTeX copied to clipboard.');
            }
        });
    });

    document.querySelectorAll('[data-github-repo]').forEach(link => {
        const repository = link.dataset.githubRepo;
        const stars = link.querySelector('.repo-stars');
        const count = link.querySelector('.repo-star-count');

        if (!repository || !stars || !count) {
            return;
        }

        const encodedRepository = repository.split('/').map(encodeURIComponent).join('/');
        fetch(`https://api.github.com/repos/${encodedRepository}`, {
            headers: { Accept: 'application/vnd.github+json' }
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`GitHub API returned ${response.status}`);
                }
                return response.json();
            })
            .then(repositoryData => {
                const starCount = repositoryData.stargazers_count;
                if (!Number.isInteger(starCount)) {
                    return;
                }

                const formattedCount = new Intl.NumberFormat('en', {
                    notation: starCount >= 1000 ? 'compact' : 'standard',
                    maximumFractionDigits: 1
                }).format(starCount);
                const label = `${starCount} GitHub ${starCount === 1 ? 'star' : 'stars'}`;

                count.textContent = formattedCount;
                stars.setAttribute('aria-label', label);
                stars.title = label;
            })
            .catch(() => {
                // Keep the server-rendered count when GitHub is unavailable or rate-limited.
            });
    });

    const alphaXivLinks = Array.from(document.querySelectorAll('[data-alphaxiv-paper]'))
        .filter(link => link.dataset.alphaxivPaper && link.querySelector('.alphaxiv-like-count'));

    const fetchWithTimeout = (url, timeoutMs = 8000) => {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
        return fetch(url, {
            headers: { Accept: 'application/json' },
            cache: 'no-store',
            signal: controller.signal
        }).finally(() => window.clearTimeout(timeout));
    };

    const updateAlphaXivLike = async link => {
        const paperId = link.dataset.alphaxivPaper;
        const count = link.querySelector('.alphaxiv-like-count');
        const encodedPaperId = encodeURIComponent(paperId);
        const cacheBuster = `?homepage_timestamp=${Date.now()}`;
        const endpoints = [
            `https://api.alphaxiv.org/papers/v3/legacy/${encodedPaperId}${cacheBuster}`,
            `https://proxy.cors.sh/https://api.alphaxiv.org/papers/v3/legacy/${encodedPaperId}${cacheBuster}`
        ];

        for (const endpoint of endpoints) {
            try {
                const response = await fetchWithTimeout(endpoint);
                if (!response.ok) {
                    continue;
                }

                const payload = await response.json();
                const likeCount = payload?.paper?.paper_group?.metrics?.public_total_votes;
                if (!Number.isInteger(likeCount) || likeCount < 0) {
                    continue;
                }

                count.textContent = String(likeCount);
                const label = `${likeCount} ${likeCount === 1 ? 'like' : 'likes'} on alphaXiv`;
                link.setAttribute('aria-label', label);
                link.title = label;
                return;
            } catch {
                // Keep the server-rendered fallback when alphaXiv is unavailable.
            }
        }
    };

    const refreshAlphaXivLikes = () => {
        if (document.visibilityState === 'hidden') {
            return;
        }
        alphaXivLinks.forEach(link => updateAlphaXivLike(link));
    };

    if (alphaXivLinks.length > 0) {
        refreshAlphaXivLikes();
        window.setInterval(refreshAlphaXivLikes, 5 * 60 * 1000);
    }

    document.addEventListener('keydown', function(event) {
        if (event.key !== 'Escape') {
            return;
        }

        document.querySelectorAll('.bib-toggle[aria-expanded="true"]').forEach(button => {
            const panel = document.getElementById(button.getAttribute('aria-controls'));
            if (panel) {
                panel.hidden = true;
            }
            button.setAttribute('aria-expanded', 'false');
        });
    });

});

function copyText(text, message) {
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text)
            .then(() => showNotification(message))
            .catch(() => copyTextWithFallback(text, message));
        return;
    }

    copyTextWithFallback(text, message);
}

function copyTextWithFallback(text, message) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    try {
        if (document.execCommand('copy')) {
            showNotification(message);
        }
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
