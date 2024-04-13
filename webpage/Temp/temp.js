const searchInput = document.getElementById("search-input");
const searchButton = document.getElementById("search-button");

searchButton.addEventListener("click", function() {
  const searchTerm = searchInput.value;
  // Add your search functionality here
  alert("Searching for " + searchTerm);
});

searchInput.addEventListener("input", function() {
  searchInput.value = searchInput.value.toUpperCase();
});