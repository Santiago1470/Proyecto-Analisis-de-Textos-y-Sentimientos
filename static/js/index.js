var comentarios = {}

let map;
let marker;

$(document).ready(function () {
    // Inicializar mapa centrado en Bogotá
    map = L.map('map').setView([4.7110, -74.0721], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    $("#buscar").on("click", () => {
        var texto = $("#textaAnalizar").val();
        let chat = document.querySelector("#textoChat");
        const userDiv = document.createElement("div");
        userDiv.className = "contenedor-usuario";
        userDiv.id = "textoUsuario";
        userDiv.textContent = texto;
        chat.appendChild(userDiv);
        scrollToBottom();

        $.ajax({
            url: 'http://localhost:3000/analizarTexto',
            type: 'POST',
            dataType: 'json',
            contentType: 'application/json',
            data: JSON.stringify({ texto: texto }),
            success: function (data) {
                console.log(data[0]);
                console.log(texto);

                const botDiv = document.createElement("div");
                botDiv.className = "contenedor-bot";
                botDiv.id = "textoBot";
                botDiv.textContent = data[1];
                chat.appendChild(botDiv);
                scrollToBottom();

                $("#textaAnalizar").val("");

                if (data[0].lugar && data[0].lugar.lat && data[0].lugar.lng) {
                    const lat = data[0].lugar.lat;
                    const lng = data[0].lugar.lng;
                    const nombre = data[0].lugar.nombre;

                    map.setView([lat, lng], 15);

                    if (marker) {
                        map.removeLayer(marker);
                    }

                    marker = L.marker([lat, lng]).addTo(map)
                        .bindPopup(`<b>${nombre}</b>`)
                        .openPopup();
                }
            },
            error: function (xhr, status, error) {
                console.error(xhr.responseText);
            }
        });
    });
});

function scrollToBottom() {
    const contenedorRespuesta = document.getElementById('textoChat');
    contenedorRespuesta.scrollTop = contenedorRespuesta.scrollHeight;
}
