/**
 * IRIS Literature Browser - Main JavaScript
 */

// Debounce utility function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Search functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.querySelector('.nav-search input[name="q"]');
    const searchForm = document.querySelector('.nav-search form');

    if (searchInput && searchForm) {
        // Auto-submit search with debouncing
        let searchTimeout;

        searchInput.addEventListener('input', debounce(function(e) {
            const query = e.target.value.trim();

            // Only auto-submit if we have a meaningful query
            if (query.length >= 3) {
                searchForm.submit();
            }
        }, 500));
    }

    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Add active state to current page in navigation
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');

    navLinks.forEach(link => {
        const linkPath = new URL(link.href).pathname;
        if (currentPath === linkPath) {
            link.style.background = 'rgba(255,255,255,0.1)';
        }
    });

    // Initialize paper card hover effects
    const paperCards = document.querySelectorAll('.paper-card');
    paperCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Handle category filter change
    const categorySelect = document.querySelector('.filters select[name="category"]');
    if (categorySelect) {
        categorySelect.addEventListener('change', function() {
            this.form.submit();
        });
    }

    // Handle sort by change
    const sortSelect = document.querySelector('.filters select[name="order_by"]');
    if (sortSelect) {
        sortSelect.addEventListener('change', function() {
            this.form.submit();
        });
    }
});

// Utility: Format date for display
function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Utility: Truncate text with ellipsis
function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Console welcome message
console.log('%cIRIS Literature Browser', 'color: #2c3e50; font-size: 20px; font-weight: bold;');
console.log('%cIntelligent Research Information System', 'color: #3498db; font-size: 14px;');
