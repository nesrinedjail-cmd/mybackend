<?php
require_once "db.php";

$data = json_decode(file_get_contents("php://input"), true);
$email = $data["email"];

$stmt = $pdo->prepare("SELECT id FROM users WHERE email = ?");
$stmt->execute([$email]);
$user = $stmt->fetch(PDO::FETCH_ASSOC);

if ($user) {
    $token = bin2hex(random_bytes(50));
    $expires = date("Y-m-d H:i:s", strtotime("+1 hour"));

    $stmt = $pdo->prepare("UPDATE users SET reset_token=?, reset_expires=? WHERE id=?");
    $stmt->execute([$token, $expires, $user['id']]);

    $reset_link = "http://yourdomain.com/reset_password.php?token=$token";

    mail($email, "Reset Password", $reset_link);

    echo json_encode(["success" => true]);
} else {
    echo json_encode(["success" => false, "message" => "Email not found"]);
}