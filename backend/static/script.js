// When the window is loaded, fetch the posts
window.onload = function() {
    fetchFilteredPosts();
};


 function createSentimentChart(data) {
        // Calculate the max score for scaling purposes
        const maxScore = Math.max(...data.map(d => d.score));

        // Create an SVG element
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '400');
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
    const subreddit = document.getElementById('subredditDropdown').value;
    const category = document.getElementById('categoryDropdown').value;
    fetch(`/api/posts?subreddit=${subreddit}&category=${category}`)
        .then(response => response.json())
        .then(data => {
            const postsSection = document.getElementById('posts');
            postsSection.innerHTML = ''; // Clear out the current content
            data.posts.forEach(post => {
                const postElement = document.createElement('article');
                const chartElement = document.createElement('div');
                postElement.className = 'post';
                postElement.innerHTML = `
                    <h2>${post.title}</h2>
                    <p>${post.content}</p>
                    <p>${post.emotion}</p>
                `;
                console.log(JSON.parse(post.emotion))
                postElement.appendChild(createSentimentChart(JSON.parse(post.emotion)[0]))
                postsSection.appendChild(postElement);
            });
        })
        .catch(error => console.error('Error fetching posts:', error));
}

