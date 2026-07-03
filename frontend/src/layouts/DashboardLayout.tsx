import { Outlet } from "react-router-dom";
import Header from "../components/Header";

export default function DashboardLayout() {
  return (
    <div className="flex min-h-screen flex-col bg-dss-bg">
      <Header />
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-dss-border bg-dss-panel px-6 py-2">
        <p className="text-[10px] text-gray-600 text-center">
          DSSPrototype &mdash; AI Decision Support System &mdash; v0.1.0
        </p>
      </footer>
    </div>
  );
}
