import sqlite3
import tkinter as tk
from tkinter import ttk, simpledialog
import queue

class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Monitoring System")
        self.tree = ttk.Treeview(root, columns=(
            "Student ID", "Status", "Time", "Last Seen Time", "Presence Status"),
                                 show='headings')

        for col in (
                "Student ID", "Status", "Time", "Last Seen Time", "Presence Status"):
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=120)

        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_double_click)

        # 새로운 메모리 캐시
        self.data_queue = queue.Queue()

    def update_table(self):
        """캐시에서 데이터를 읽어와 GUI를 업데이트"""
        for i in self.tree.get_children():
            try:
                self.tree.delete(i)
            except tk.TclError:
                # If item is already deleted, skip it
                continue

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance ORDER BY student_id ASC")
        rows = cursor.fetchall()
        for row in rows:
            self.tree.insert("", tk.END, values=row)
        conn.close()

    # 프로그램 종료 후 출결 상태를 수동으로 변경하기 위한 함수
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
        except IndexError:
            pass
