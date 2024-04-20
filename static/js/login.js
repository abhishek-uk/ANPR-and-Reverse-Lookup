const users = [
    {
        username: 'albin',
        password: 'albin@123'
    },
    {
        username: 'nasla',
        password: 'nasla@123'
    }
]

const admins = [
    {
        username: 'zoomi',
        password: 'zoomi@123'
    }
]



function login(){
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const userRadio = document.getElementById('user');
    const adminRadio = document.getElementById('admin');

    if (userRadio.checked) {
        for (const user of users) {
            if (user.username === username && user.password === password) {
                window.location.href = '/user-home.html';  
                // window.location.href = "{{ url_for('static', filename='user-home.html') }}";
                return;
            }
        }
    } else if (adminRadio.checked) {
        for (const admin of admins) {
            if (admin.username === username && admin.password === password) {
                window.location.href = 'admin-home.html';   
                // window.location.href = "{{ url_for('static', filename='admin-home.html') }}";
                return;
            }
        }
    }

    if (username.trim() === "")
        alert('Username cannot be blank')
    else if(password.trim() == "")
        alert('Password cannot be blank')
    else
        alert('Invalid username or password.')
}