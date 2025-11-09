// Navigation Toggle for Mobile
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.getElementById('nav-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (navToggle) {
        navToggle.addEventListener('click', function() {
            navLinks.classList.toggle('nav-open');
            navToggle.classList.toggle('active');
        });
    }

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    // Observe all sections (except no-animation sections)
    document.querySelectorAll('.section').forEach(section => {
        if (!section.classList.contains('no-animation')) {
            section.classList.add('fade-in');
            observer.observe(section);
        }
    });

    // Active navigation highlight
    function updateActiveNav() {
        const sections = document.querySelectorAll('.section');
        const navLinks = document.querySelectorAll('.nav-links a');
        
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (window.pageYOffset >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').substring(1) === current) {
                link.classList.add('active');
            }
        });
    }

    // Update active nav on scroll
    window.addEventListener('scroll', updateActiveNav);
    updateActiveNav(); // Initial call

    // Visitor counter (simulated)
    function updateVisitorCounter() {
        // This is a simple simulation. In a real application, you'd use a backend service
        const pageViews = localStorage.getItem('pageViews') || 0;
        const todayVisitors = localStorage.getItem('todayVisitors') || 0;
        
        // Increment counters
        localStorage.setItem('pageViews', parseInt(pageViews) + 1);
        
        // Check if it's a new day
        const today = new Date().toDateString();
        const lastVisit = localStorage.getItem('lastVisit');
        
        if (lastVisit !== today) {
            localStorage.setItem('todayVisitors', 1);
            localStorage.setItem('lastVisit', today);
        } else {
            localStorage.setItem('todayVisitors', parseInt(todayVisitors) + 1);
        }
        
        // Update display
        const pageViewsElement = document.getElementById('page-views');
        const todayVisitorsElement = document.getElementById('today-visitors');
        
        if (pageViewsElement) {
            pageViewsElement.textContent = localStorage.getItem('pageViews');
        }
        
        if (todayVisitorsElement) {
            todayVisitorsElement.textContent = localStorage.getItem('todayVisitors');
        }
    }

    // Initialize visitor counter
    setTimeout(updateVisitorCounter, 1000);

    // Add hover effects to project cards
    const projectCards = document.querySelectorAll('.project-card');
    projectCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Typing effect removed - name should display immediately

    // Parallax effect removed to ensure immediate display of hero section

    // Email protection (simple obfuscation)
    const emailLinks = document.querySelectorAll('a[href*="mailto"]');
    emailLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // You can add additional email protection logic here
            console.log('Email link clicked');
        });
    });

    // Add loading animation
    function showLoading() {
        document.body.classList.add('loading');
    }

    function hideLoading() {
        document.body.classList.remove('loading');
    }

    // Simulate loading
    showLoading();
    window.addEventListener('load', function() {
        setTimeout(hideLoading, 500);
    });

    // Add scroll to top functionality
    function addScrollToTop() {
        const scrollButton = document.createElement('button');
        scrollButton.innerHTML = '↑';
        scrollButton.className = 'scroll-to-top';
        scrollButton.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            font-size: 20px;
            cursor: pointer;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 1001;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        `;
        
        document.body.appendChild(scrollButton);
        
        scrollButton.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
        
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                scrollButton.style.opacity = '1';
                scrollButton.style.visibility = 'visible';
            } else {
                scrollButton.style.opacity = '0';
                scrollButton.style.visibility = 'hidden';
            }
        });
    }

    // Initialize scroll to top
    addScrollToTop();

    // Add copy to clipboard functionality for email
    function addCopyToClipboard() {
        const emailElement = document.querySelector('a[href="mailto:haoyuntang224@163.com"]');
        if (emailElement) {
            emailElement.addEventListener('click', function(e) {
                e.preventDefault();
                const email = 'haoyuntang224@163.com';
                
                if (navigator.clipboard) {
                    navigator.clipboard.writeText(email).then(function() {
                        showNotification('Email address copied to clipboard!');
                    });
                } else {
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = email;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    showNotification('Email address copied to clipboard!');
                }
            });
        }
    }

    // Show notification
    function showNotification(message) {
        const notification = document.createElement('div');
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: #4ade80;
            color: white;
            padding: 1rem 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            z-index: 1002;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    // Initialize copy functionality
    addCopyToClipboard();

    console.log('Academic homepage initialized successfully! �');
});

// Add enhanced animations and effects
function addEnhancedAnimations() {
    // Parallax effect disabled for hero section to ensure immediate display

    // Stagger animation for project cards
    const projectCards = document.querySelectorAll('.project-card');
    projectCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('stagger-animation');
    });

    // Icon hover effects
    document.querySelectorAll('.social-links a, .project-link').forEach(link => {
        link.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px) scale(1.05)';
        });
        
        link.addEventListener('mouseleave', function() {
            this.style.transform = '';
        });
    });

    // Floating animation for profile image
    const profileImg = document.querySelector('.profile-image img');
    if (profileImg) {
        profileImg.style.animation = 'float 6s ease-in-out infinite';
    }

    // Add cursor trail effect
    let cursorTrail = [];
    document.addEventListener('mousemove', (e) => {
        cursorTrail.push({ x: e.clientX, y: e.clientY });
        if (cursorTrail.length > 20) cursorTrail.shift();
    });
}

// Add CSS for enhanced animations
const enhancedStyle = document.createElement('style');
enhancedStyle.textContent = `
    .loading * {
        transition: none !important;
        animation-duration: 0s !important;
    }
    
    .nav-links.nav-open {
        display: flex;
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.98);
        flex-direction: column;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(15px);
    }
    
    .nav-toggle.active span:nth-child(1) {
        transform: rotate(-45deg) translate(-5px, 6px);
    }
    
    .nav-toggle.active span:nth-child(2) {
        opacity: 0;
    }
    
    .nav-toggle.active span:nth-child(3) {
        transform: rotate(45deg) translate(-5px, -6px);
    }
    
    .nav-links a.active {
        color: #667eea;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes staggerFadeIn {
        from {
            opacity: 0;
            transform: translateY(30px) scale(0.9);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .stagger-animation {
        animation: staggerFadeIn 0.6s ease-out forwards;
    }
    
    @media (max-width: 768px) {
        .nav-links {
            display: none;
        }
    }
`;
document.head.appendChild(enhancedStyle);

// Initialize enhanced animations
addEnhancedAnimations();

// Auto-update footer last updated text if element exists
(() => {
    const footerText = document.querySelector('#footer-text p');
    if (footerText) {
        const now = new Date();
        const monthNames = ['January','February','March','April','May','June','July','August','September','October','November','December'];
        const formatted = `${monthNames[now.getMonth()]} ${now.getFullYear()}`;
        footerText.innerHTML = `© ${now.getFullYear()} Haoyun Tang | Last updated: ${formatted}`;
    }
})();



// Email copy functionality with enhanced feedback
function copyEmail(event) {
    event.preventDefault();
    const email = 'haoyuntang224@163.com';
    const emailElement = event.target;
    
    // Add temporary visual feedback
    const originalText = emailElement.textContent;
    emailElement.style.transition = 'all 0.3s ease';
    emailElement.style.color = '#22c55e';
    emailElement.textContent = '✓ Copied!';
    
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(email).then(() => {
            showNotification('📧 Email copied to clipboard!', 'success');
            
            // Reset after animation
            setTimeout(() => {
                emailElement.style.color = '';
                emailElement.textContent = originalText;
            }, 1500);
        });
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = email;
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
            document.execCommand('copy');
            showNotification('📧 Email copied to clipboard!', 'success');
            
            setTimeout(() => {
                emailElement.style.color = '';
                emailElement.textContent = originalText;
            }, 1500);
        } catch (err) {
            showNotification('❌ Failed to copy email', 'error');
            emailElement.style.color = '';
            emailElement.textContent = originalText;
        }
        document.body.removeChild(textArea);
    }
}