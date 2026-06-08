# 🍽 CampusBite

A campus food ordering platform — a single-page web app with three role-based portals: **Student**, **Vendor**, and **Admin**.

## 🚀 Live Demo

Open `index.html` in any modern browser — no server or installation needed.

## 📁 Project Structure

```
campusbite/
├── index.html        # Main HTML (entry point)
├── css/
│   └── style.css     # All styles & responsive design
├── js/
│   └── app.js        # All JavaScript logic
└── README.md
```

## ✨ Features

### 🎒 Student Portal
- Browse today's menu with category filters (Meals, Snacks, Drinks, Desserts)
- Add items to cart with quantity controls
- View invoice with platform fee breakdown
- Select payment method: UPI, Card, or Cash on Delivery
- Choose a pickup/delivery time slot
- Get a unique order code (e.g. `CB-847291`) on confirmation

### 🍳 Vendor Portal
- Look up any order by its code
- View all live (non-completed) orders
- Update order status: New → Preparing → Ready → Done
- Cancel orders

### ⚙️ Admin Panel
- Dashboard stats: Total Orders, Revenue, Menu Items, Platform Fees
- View & manage full menu (Edit, Hide/Show, Delete items)
- Add new menu items with image URL, category, type & badge
- View all orders across all statuses

## 🛠 Tech Stack

- Pure HTML, CSS, JavaScript — zero dependencies, zero build step
- Google Fonts: [Syne](https://fonts.google.com/specimen/Syne) + [Plus Jakarta Sans](https://fonts.google.com/specimen/Plus+Jakarta+Sans)
- Food images via [Unsplash](https://unsplash.com)

## 📦 How to Upload to GitHub

1. Create a new repository on [github.com](https://github.com/new)
2. Clone it locally: `git clone https://github.com/YOUR_USERNAME/campusbite.git`
3. Copy all files from this folder into the cloned repo
4. Run:
   ```bash
   git add .
   git commit -m "Initial commit — CampusBite"
   git push origin main
   ```
5. Enable **GitHub Pages** under Settings → Pages → Branch: `main` → Save
6. Your site will be live at `https://YOUR_USERNAME.github.io/campusbite/`

## 📝 License

MIT — free to use, modify and deploy.
