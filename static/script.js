const input = document.getElementById('pokemon_image');
const preview = document.getElementById('previewImg');
const predictBtn = document.getElementById('predictBtn');

input.addEventListener('change', function() {
    const file = this.files[0];
    if(file){
        const reader = new FileReader();
        reader.onload = function(e){
            preview.src = e.target.result;
            preview.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

predictBtn.addEventListener('click', function(){
    if(!input.files[0]){
        alert("Please choose a Pokémon image first!");
        return;
    }
    alert("Ready to send this image to your CNN model!");
});const fileInput = document.getElementById("pokemon_image");
const previewImg = document.getElementById("previewImg");
const pokeball = document.querySelector(".pokeball");

