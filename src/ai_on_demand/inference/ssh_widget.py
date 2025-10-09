from typing import Optional
from pathlib import Path

import napari
import paramiko 
from qtpy.QtWidgets import (
    QWidget,
    QGridLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLayout
)
from ai_on_demand.widget_classes import SubWidget


class SshWidget(SubWidget):
    _name = "ssh"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            title="SSH",
            parent=parent,
            layout=layout,
            tooltip="Run pipelines on nemo from local",
            **kwargs,
        )

    def create_box(self, variant: Optional[str] = None):
        # --- Input fields ---
        self.command_input = QLineEdit(placeholderText="Enter command here...")
        self.hostname = QLineEdit(placeholderText="Enter hostname here...")
        self.username = QLineEdit(placeholderText="Enter username here...")
        self.passphrase_input = QLineEdit(placeholderText="Enter passphrase here...")
        self.passphrase_input.setEchoMode(QLineEdit.Password)

        # --- SSH key section ---
        self.info_btn = QPushButton("i")
        self.info_btn.setFixedWidth(30)
        self.info_btn.setToolTip("Help I don't know which ssh key to pick!")
        self.info_btn.clicked.connect(self.show_ssh_info)

        self.ssh_key_path = ""
        self.ssh_key_label = QLabel("SSH Key: Not selected")
        self.locate_key_btn = QPushButton("Locate SSH Key")
        self.locate_key_btn.clicked.connect(self.locate_ssh_key)

        # --- Send command button ---
        self.send_btn = QPushButton("SEND COMMAND")
        self.send_btn.clicked.connect(self._run_command)

        # --- Layout setup ---
        layout = QGridLayout()

        # Row 0: Command input
        layout.addWidget(QLabel("Command:"), 0, 0)
        layout.addWidget(self.command_input, 0, 1, 1, 2)

        # Row 1: Hostname
        layout.addWidget(QLabel("Hostname:"), 1, 0)
        layout.addWidget(self.hostname, 1, 1, 1, 2)

        # Row 2: Username
        layout.addWidget(QLabel("Username:"), 2, 0)
        layout.addWidget(self.username, 2, 1, 1, 2)

        # Row 3: Passphrase
        layout.addWidget(QLabel("Passphrase:"), 3, 0)
        layout.addWidget(self.passphrase_input, 3, 1, 1, 2)

        # Row 4: SSH Key label
        layout.addWidget(self.ssh_key_label, 4, 0, 1, 3)

        # Row 5: SSH Key buttons (Locate + Info)
        layout.addWidget(self.locate_key_btn, 5, 0, 1, 2)
        layout.addWidget(self.info_btn, 5, 2)

        # Row 6: Send command button
        layout.addWidget(self.send_btn, 6, 0, 1, 3)

        # self.setLayout(layout)
        self.inner_layout.addLayout(layout,0 , 0, 1, 1)


    def locate_ssh_key(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Key", str(Path.home() / ".ssh")
        )
        if file_path:
            self.ssh_key_path = file_path
            self.ssh_key_label.setText(f"SSH Key: {file_path}")

    def show_ssh_info(self):
        QMessageBox.information(
            self,
            "SSH Key Help",
            (
                "For NEMO, you likely need to select your RSA key (id_rsa) in .ssh from your home directory.\n\n"
                "If you can't see hidden folders you may need to toggle visibility:\n"
                "macOS: Command + Shift + .\n"
                "Linux: Ctrl + H (may differ by distro)\n"
                "Windows: Enable 'Hidden items' in the View menu."
            ),
        )
    
    def _run_ssh(self, command: str, hostname: str, username: str, passphrase: str, ssh_key_path):
        # key_path = Path.home() / ".ssh/id_ed25519"
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy
        )  # maybe worth getting the known hosts from the known_hosts file in the .ssh dir

        ssh.connect(
            hostname=hostname,
            username=username,
            key_filename=str(ssh_key_path),
            passphrase=passphrase,
        )

        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode()
        ssh.close()
        return output


    def _run_command(self):
        command = self.command_input.text()
        hostname = self.hostname.text()
        username = self.username.text()
        passphrase = self.passphrase_input.text()
        print(
            self._run_ssh(command, hostname, username, passphrase, self.ssh_key_path)
        )
