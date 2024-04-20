if (document.getElementById('from_vid_upload')){
  for(let i = 1; i <= 4; i++){
    document.getElementById('admin-sec-' + i).style.display = 'none';
    document.getElementById('admin-button-' + i).classList.remove('active')
  }
  document.getElementById('admin-sec-2').style.display = 'block';
  document.getElementById('admin-button-2').classList.add('active');
  document.getElementById('vid-upload-result').style.display = 'flex';



}else{
  for(let i = 1; i <= 4; i++){
    document.getElementById('admin-sec-' + i).style.display = 'none';
    document.getElementById('admin-button-' + i).classList.remove('active')
  }
  document.getElementById('admin-sec-1').style.display = 'block';
  document.getElementById('admin-button-1').classList.add('active');
}




function ChangeWorkSection(divNum){
  for(let i = 1; i <= 4; i++){
    document.getElementById('admin-sec-' + i).style.display = 'none';
    document.getElementById('admin-button-' + i).classList.remove('active')
  }
  document.getElementById(('admin-sec-' + divNum)).style.display = 'block';
  document.getElementById(('admin-button-' + divNum)).classList.add('active');
}


function load_animation(){
  document.getElementById('loader-container').style.display = 'flex'
}