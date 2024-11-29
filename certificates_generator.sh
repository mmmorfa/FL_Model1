# Generate root certificate (CA)
openssl genpkey -algorithm RSA -out Certificates/rootCA.key
openssl req -x509 -new -nodes -key Certificates/rootCA.key -subj "/C=ES/ST=BCN/L=Castelldefels/O=UPC/OU=ENTEL/CN=CA" -sha256 -days 365 -out Certificates/rootCA.pem

# Generate server private key
openssl genpkey -algorithm RSA -out Certificates/server.key

# Generate a certificate signing request (CSR) for the server
openssl req -new -key Certificates/server.key -subj "/C=ES/ST=BCN/L=Castelldefels/O=UPC/OU=ENTEL/CN=localhost" -out Certificates/server.csr

# Sign the server certificate with the root certificate
openssl x509 -req -in Certificates/server.csr -CA Certificates/rootCA.pem -CAkey Certificates/rootCA.key -CAcreateserial -out Certificates/server.pem -days 365 -sha256
