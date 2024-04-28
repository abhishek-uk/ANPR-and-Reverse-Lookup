if (document.getElementById('from_vid_upload')){
  for(let i = 1; i <= 4; i++){
    document.getElementById('admin-sec-' + i).style.display = 'none';
    document.getElementById('admin-button-' + i).classList.remove('active')
  }
  document.getElementById('admin-sec-2').style.display = 'block';
  document.getElementById('admin-button-2').classList.add('active');
  document.getElementById('vid-upload-result').style.display = 'flex';


}else if (document.getElementById('from_db_display')){
  for(let i = 1; i <= 4; i++){
    document.getElementById('admin-sec-' + i).style.display = 'none';
    document.getElementById('admin-button-' + i).classList.remove('active')
  }
  document.getElementById('admin-sec-4').style.display = 'block';
  document.getElementById('admin-button-4').classList.add('active');

  const array = JSON.parse(jsArray);

  for (let i = 0; i < array.length; i++) {
    const subArray = array[i];
    addRow("rec-vehicle-table", subArray)
  }
  

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

function addRow(tableID, rowData) {
  var table = document.getElementById(tableID);
  var newRow = table.insertRow();

  for (var i = 0; i < rowData.length; i++) {
    var cell = newRow.insertCell(i);
    cell.innerHTML = rowData[i];
  }
}