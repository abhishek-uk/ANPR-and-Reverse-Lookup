
function ChangeWorkSection(divNum){
    for(let i = 1; i <= 4; i++){
      document.getElementById('admin-sec-' + i).style.display = 'none';
      document.getElementById('admin-button-' + i).classList.remove('active')
    }
    document.getElementById(('admin-sec-' + divNum)).style.display = 'block';
    document.getElementById(('admin-button-' + divNum)).classList.add('active');
  }