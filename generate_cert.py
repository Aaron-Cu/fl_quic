import ipaddress
import datetime
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from cryptography.hazmat.backends import default_backend

# 1. Create CA private key
ca_private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

# 2. Create CA cert
ca_subject = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"My Test CA"),
    x509.NameAttribute(NameOID.COMMON_NAME, u"My Test Root CA"),
])

ca_cert = x509.CertificateBuilder(
).subject_name(
    ca_subject
).issuer_name(
    ca_subject
).public_key(
    ca_private_key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.datetime.utcnow()
).not_valid_after(
    datetime.datetime.utcnow() + datetime.timedelta(days=3650)
).add_extension(
    x509.BasicConstraints(ca=True, path_length=None),
    critical=True,
).sign(ca_private_key, hashes.SHA256(), default_backend())

# Write CA private key and cert
with open("ca_key.pem", "wb") as f:
    f.write(ca_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ))

with open("ca_cert.pem", "wb") as f:
    f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

print("✅ CA cert and key generated.")

# 3. Create server private key
server_private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

# 4. Create server CSR
server_subject = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Georgia"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, u"Localhost"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Test Server"),
    x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
])

alt_names = x509.SubjectAlternativeName([
    x509.DNSName(u"localhost"),
    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
])

server_cert = x509.CertificateBuilder(
).subject_name(
    server_subject
).issuer_name(
    ca_subject
).public_key(
    server_private_key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.datetime.utcnow()
).not_valid_after(
    datetime.datetime.utcnow() + datetime.timedelta(days=365)
).add_extension(
    alt_names,
    critical=False,
).sign(ca_private_key, hashes.SHA256(), default_backend())

# Write server key and cert
with open("key.pem", "wb") as f:
    f.write(server_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ))

with open("cert.pem", "wb") as f:
    f.write(server_cert.public_bytes(serialization.Encoding.PEM))

print("✅ Server cert and key signed by CA generated.")
