// When the window is loaded, fetch the posts
window.onload = function () {
    fetchFilteredPosts();
};


function createSentimentChart(data) {
    // Calculate the max score for scaling purposes
    const maxScore = Math.max(...data.map(d => d.score));

    // Create an SVG element
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '2000');
    svg.setAttribute('height', '200');


    data.forEach((d, i) => {
        // Create a group for each bar to hold the rect and text
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        // Create the rectangle for the bar
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        const barHeight = (d.score / maxScore) * 200; // Scale the bar height to the SVG height
        rect.setAttribute('width', '40');
        rect.setAttribute('height', barHeight);
        rect.setAttribute('x', i * 50 + 30); // 50 units apart, plus 30 units from the left edge
        rect.setAttribute('y', 200 - barHeight); // SVG y starts at the top
        rect.setAttribute('fill', 'teal');

        // Create the text label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.textContent = d.label;
        text.setAttribute('x', i * 50 + 50); // Center the text under the bar
        text.setAttribute('y', 195); // Just above the bottom of the SVG
        text.setAttribute('font-size', '10');
        text.setAttribute('text-anchor', 'middle');

        // Append the rect and text to the group
        group.appendChild(rect);
        group.appendChild(text);

        // Append the group to the SVG
        svg.appendChild(group);
    });

    return svg;
}

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