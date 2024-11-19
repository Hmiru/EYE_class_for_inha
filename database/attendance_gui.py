import sqlite3
import tkinter as tk
from tkinter import ttk, simpledialog

class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Monitoring System")
        self.tree = ttk.Treeview(root, columns=(
            "Student ID", "Status", "Time", "Last Seen Time", "Presence Status", "Recent Focus", "Cumulative Focus"),
                                 show='headings')

        for col in (
                "Student ID", "Status", "Time", "Last Seen Time", "Presence Status", "Recent Focus", "Cumulative Focus"):
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=120)

        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_double_click)

    def update_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance")
        rows = cursor.fetchall()
        for row in rows:
            self.tree.insert("", tk.END, values=row)
        conn.close()

    def on_double_click(self, event):
        try:
            item = self.tree.selection()[0]
            student_id = self.tree.item(item, "values")[0]
            current_status = self.tree.item(item, "values")[1]
            new_status = simpledialog.askstring("Update Status", f"Enter new status for Student ID {student_id} (current: {current_status}):")
            if new_status:
                conn = sqlite3.connect("attendance.db")
                cursor = conn.cursor()
                cursor.execute("UPDATE attendance SET status = ? WHERE student_id = ?", (new_status, student_id))
                conn.commit()
                conn.close()
                self.update_table()
        except IndexError:
            pass
