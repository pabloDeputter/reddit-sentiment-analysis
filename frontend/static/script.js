// When the window is loaded, fetch the posts
window.onload = function () {
    fetchFilteredPosts();
};

document.addEventListener('DOMContentLoaded', function () {
    const input = document.getElementById('subredditInput');
    const suggestionsContainer = document.getElementById('suggestions');
    let debounceTimer;

    input.addEventListener('input', function () {
        clearTimeout(debounceTimer);
        const query = input.value;

        debounceTimer = setTimeout(() => {
            if (!query) {
                suggestionsContainer.innerHTML = '';
                return;
            }

            fetch(`/api/subreddits?query=${encodeURIComponent(query)}}`)
                .then(response => response.json())
                .then(data => {
                    suggestionsContainer.innerHTML = '';
                    data.forEach(subreddit => {
                        const div = document.createElement('div');
                        div.classList.add('suggestion-item');
                        div.textContent = subreddit;
                        div.onclick = function () {
                            input.value = subreddit;
                            suggestionsContainer.innerHTML = '';
                        };
                        suggestionsContainer.appendChild(div);
                    });
                })
                .catch(error => console.error('Error fetching suggestions:', error));
        }, 250); // Adjust the delay (in milliseconds) as needed
    });
});


function fetchFilteredPosts() {
    let subreddit = document.getElementById('subredditInput').value;
    if (subreddit === '') {
        subreddit = 'unpopularopinion'
    }

    const emotion = document.getElementById('categoryDropdown').value;
    fetch(`/api/posts?subreddit=${subreddit}&emotion=${emotion}`)
        .then(response => response.json())
        .then(data => {
            const postsSection = document.getElementById('posts');
            postsSection.innerHTML = ''; // Clear out the current content
            data.posts.forEach((post, index) => {
                const postElement = document.createElement('article');
                const chartId = `emotionChart-${index}`; // Unique ID for each chart
                postElement.className = 'post';

                // Retrieve and process data
                const emotions = JSON.parse(post.emotion);


                // Sort and take top 3 emotions
                let topEmotions = [...emotions];
                topEmotions = topEmotions.sort((a, b) => b.score - a.score);
                topEmotions = topEmotions.slice(0, 3);

                // Create top emotions text
                const topEmotionsText = topEmotions.map(e => e.score ? `${e.label}: ${e.score.toFixed(3)}` : `${e.label}: N/A`).join(', ');
                postElement.innerHTML = `
                    <div class="post-content">
                        <h2>${post.title}</h2>
                        <p>${post.content}</p>
                        <p id="top-emotions" class="top-emotions">${topEmotionsText}</p>
                        ${emotion === 'all' ? '' : `<p class="top-emotions">${emotion}: ${emotions.find(item => item.label === emotion).score.toFixed(3)}</p>`}
                    </div>
                    <div class="divider"/>
                `;
                postsSection.appendChild(postElement);


                const colors = ['#FF6384', '#36A2EB', '#d30008', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF', '#7ACB80', '#C2847A', '#FA8072', '#8A2BE2', '#5F9EA0', '#FF4500', '#2E8B57', '#D2691E', '#FFD700', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#b548d7', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#818181'];

                // // Inside your data processing loop
                // const voteCountElement = document.createElement('div');
                // voteCountElement.className = 'vote-count';
                //
                // // Replace these with actual data from your API
                // const upvotes = 123;
                // const downvotes = 45;
                //
                // voteCountElement.innerHTML = `
                // <span class="upvote"><i class="fa fa-arrow-up"></i> ${upvotes}</span>
                // <span class="downvote"><i class="fa fa-arrow-down"></i> ${downvotes}</span>`;
                //
                // postElement.appendChild(voteCountElement);

                // Create chart
                const chartWrapper = document.createElement('div');
                chartWrapper.className = 'chart-wrapper';
                chartWrapper.innerHTML = `<div style="" class="chart-container"><canvas id="${chartId}"></canvas></div>`;
                postElement.appendChild(chartWrapper);


                // const chartContainer = document.createElement('div');
                // chartContainer.className = 'chart-container';
                // chartContainer.innerHTML = `<canvas id="${chartId}"></canvas>`;
                // postElement.appendChild(chartContainer);
                const ctx = document.getElementById(chartId).getContext('2d');
                new Chart(ctx, {
                    type: 'pie', data: {
                        labels: emotions.map(item => item.label), datasets: [{
                            label: 'Score',
                            data: emotions.map(item => item.score),
                            backgroundColor: colors.slice(0, emotions.map(item => item.label).length),
                            borderWidth: 1,
                            borderColor: '#fff', // White border for a cleaner look
                            hoverOffset: 5 // Slightly enlarge slice on hover for better interaction
                        }]
                    }, options: {
                        responsive: true, maintainAspectRatio: false, plugins: {
                            legend: {
                                display: true,
                                position: 'bottom',
                                labels: {
                                    usePointStyle: true,
                                    boxWidth: 10,
                                    padding: 10,
                                    fontSize: 10
                                }
                            },
                        },
                        // aspectRatio: 1.5,
                        layout: {
                            padding: {
                                top: 15, bottom: 15, left: 15, right: 15
                            }
                        }
                    }
                });


            });
        })
        .catch(error => console.error('Error fetching posts:', error));
}


const handleSubmit = (event) => {
    event.preventDefault();
    const searchValue = document.getElementById('search').value;
    // Additional actions here
}