const searchInput = document.getElementById("search-input");
const searchButton = document.getElementById("search-button");

searchButton.addEventListener("click", function() {
  const searchTerm = searchInput.value;
  Number_plate = searchTerm.replace(/\s/g, '')
  window.location.href = '/search/' + Number_plate;
});

searchInput.addEventListener("input", function() {
  searchInput.value = searchInput.value.toUpperCase();
});


var video_div = document.getElementById('vid_db');
    
if(video_div) {
    video_div.style.display = 'block';
} else {
    video_div.style.display = 'none';
}



document.addEventListener("DOMContentLoaded", function() {
  var video = document.getElementById("myVideo");
  video.addEventListener("loadedmetadata", function() {
    this.currentTime = parseInt(vid_start_time); // Start time in seconds (3 minutes 20 seconds)
  });
  video.addEventListener("timeupdate", function() {
    if (this.currentTime >= parseInt(vid_start_time) + 15) { // End time in seconds (4 minutes 30 seconds)
      this.pause();
    }
  });
});
