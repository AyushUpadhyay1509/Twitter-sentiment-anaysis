function analyzeTweet() {
    let text = document.getElementById("tweetInput").value;

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("emotionResult").innerText = data.emotion;
        document.getElementById("keywordResult").innerText = data.keywords;
    });
}

function fetchLiveTweets() {
    let keyword = document.getElementById("keywordInput").value;

    fetch(`/realtime?keyword=${keyword}`)
    .then(response => response.json())
    .then(data => {
        let tweetList = document.getElementById("tweetsList");
        tweetList.innerHTML = "";
        data.forEach(tweet => {
            let li = document.createElement("li");
            li.innerText = `${tweet.text} - Emotion: ${tweet.emotion}`;
            tweetList.appendChild(li);
        });
    });
}
