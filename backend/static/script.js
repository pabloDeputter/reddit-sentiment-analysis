document.getElementById('filterForm').addEventListener('submit', function(event) {
    console.log('prevent')
    event.preventDefault(); // Prevents default form submission
    fetchFilteredPosts(); // Call function to handle the data
});

function fetchFilteredPosts() {
    const subreddit = document.getElementById('subredditInput').value;
    const category = document.getElementById('categoryDropdown').value;
    fetch(`/api/posts?subreddit=${subreddit}&category=${category}`)
        .then(response => response.json())
        .then(data => {
            const postsSection = document.getElementById('posts');
            postsSection.innerHTML = ''; // Clear out the current content
            data.posts.forEach((post, index) => {
                const postElement = document.createElement('article');
                const chartId = `emotionChart-${index}`; // Unique ID for each chart
                postElement.className = 'post';
                postElement.innerHTML = `
                    <h2>${post.title}</h2>
                    <p>${post.content}</p>
                    <p id="top-emotions" class="top-emotions"></p>
                    <div style="width: 70%; margin: auto;"><canvas id="${chartId}"></canvas></div>
                `;
                postsSection.appendChild(postElement);

                // Retrieve and process data
                const emotions = JSON.parse(post.emotion)[0];
                // Sort and take top 3 emotions
                let topEmotions = [...emotions];
                topEmotions.sort((a, b) => b.score - a.score);
                topEmotions = topEmotions.slice(0, 3);

                const labels = topEmotions.map(item => item.label);
                const scores = topEmotions.map(item => item.score);

                // Create top emotions text
                const topEmotionsText = topEmotions.map(e => e.score ? `${e.label}: ${e.score.toFixed(2)}` : `${e.label}: N/A`).join(', ');

                // Add top emotions to post
                const topEmotionsElement = document.getElementById('top-emotions')
                // change it
                topEmotionsElement.textContent = `Top Emotions: ${topEmotionsText}`;

                // const emotionsTextElement = document.createElement('p');
                // emotionsTextElement.className = 'top-emotions';
                // emotionsTextElement.textContent = `Top Emotions: ${topEmotionsText}`;
                // postElement.appendChild(emotionsTextElement);

                const colors = ['#FF6384', '#36A2EB', '#d30008', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF', '#7ACB80', '#C2847A', '#FA8072', '#8A2BE2', '#5F9EA0', '#FF4500', '#2E8B57', '#D2691E', '#FFD700', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#b548d7', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#818181'];

                // Create chart
                const ctx = document.getElementById(chartId).getContext('2d');
                new Chart(ctx, {
                    type: 'pie', data: {
                        labels: emotions.map(item => item.label), datasets: [{
                            label: 'Emotion Scores',
                            data: emotions.map(item => item.score),
                            backgroundColor: colors.slice(0, emotions.map(item => item.label).length),
                            borderWidth: 1,
                            borderColor: '#fff', // White border for a cleaner look
                            hoverOffset: 5 // Slightly enlarge slice on hover for better interaction
                        }]
                    }, options: {
                        responsive: true, maintainAspectRatio: false, plugins: {
                            legend: {
                                position: 'bottom', labels: {
                                    boxWidth: 22, padding: 20
                                }
                            },
                        }, layout: {
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
    console.log('Submitted value:', searchValue);
    // Additional actions here
}