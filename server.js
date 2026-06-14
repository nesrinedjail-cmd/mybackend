import express from 'express';
import cors from 'cors';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

const app = express();
const PORT = process.env.PORT || 3001;
const SECRET_KEY = process.env.SECRET_KEY || 'your-secret-key-2024';

// ✅ 1. إعدادات CORS الصحيحة لمنع الخطأ
app.use(cors({
  origin: '*', // يسمح لجميع النطاقات أثناء التطوير
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

app.use(express.json());

// ✅ 2. قاعدة بيانات مؤقتة في الذاكرة (ستعود للصفر عند إعادة التشغيل)
let users = [
  {
    id: 1,
    name: "مدير النظام",
    email: "admin@example.com",
    password: "$2a$10$CwzI50peRC5pLzLwQXvnq.BZuwiQp7I4c3Z/K2I3Nt19yQMJu1QPi", // كلمة السر: admin123
    role: "admin",
    isActive: true,
    createdAt: new Date().toISOString(),
    lastLogin: null
  }
];
let videos = [];
let feedback = [];
let logs = [];

// ========== Routes ==========

// تسجيل مستخدم جديد
app.post('/api/register', async (req, res) => {
  try {
    const { name, email, password, subscription_plan = 'Free' } = req.body;
    
    if (users.find(u => u.email === email)) {
      return res.status(400).json({ error: 'البريد الإلكتروني مسجل بالفعل' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = {
      id: users.length + 1,
      name,
      email,
      password: hashedPassword,
      role: 'client',
      subscription_plan,
      subscription_status: 'Active',
      isActive: true,
      createdAt: new Date().toISOString(),
      lastLogin: null
    };

    users.push(newUser);
    const token = jwt.sign({ id: newUser.id, email, role: newUser.role }, SECRET_KEY, { expiresIn: '7d' });

    res.status(201).json({
      success: true,
      token,
      user: { id: newUser.id, name, email, role: 'client', subscription_plan }
    });
  } catch (error) {
    res.status(500).json({ error: 'حدث خطأ أثناء التسجيل' });
  }
});

// تسجيل الدخول
app.post('/api/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = users.find(u => u.email === email);

    if (!user) {
      return res.status(401).json({ error: 'البريد الإلكتروني أو كلمة المرور غير صحيحة' });
    }

    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) {
      return res.status(401).json({ error: 'البريد الإلكتروني أو كلمة المرور غير صحيحة' });
    }

    user.lastLogin = new Date().toISOString();
    logs.push({ id: logs.length + 1, action: 'تسجيل دخول', userId: user.id, timestamp: new Date().toISOString() });

    const token = jwt.sign({ id: user.id, email, role: user.role }, SECRET_KEY, { expiresIn: '7d' });

    res.json({
      success: true,
      token,
      user: { id: user.id, name: user.name, email: user.email, role: user.role }
    });
  } catch (error) {
    res.status(500).json({ error: 'حدث خطأ في الخادم' });
  }
});

// جلب الإحصائيات
app.get('/api/stats', (req, res) => {
  res.json({
    totalUsers: users.length,
    activeUsers: users.filter(u => u.lastLogin).length,
    totalVideos: videos.length,
    totalFeedback: feedback.length
  });
});

// لوحة الإيرادات (للتجربة)
app.get('/api/revenue-dashboard', (req, res) => {
  res.json({
    totalMonthlyRevenue: users.length * 10,
    activeSubscribersCount: users.filter(u => u.role === 'client').length,
    recentRenewals: [],
    subscriptionMix: [],
    revenueTrend: []
  });
});

// تشغيل الخادم
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Server running on port ${PORT}`);
});