"""add patient chat tables

Revision ID: a1c2f3d4e5f6
Revises: 5d10b4e3689f
Create Date: 2026-10-23

患者端「智慧医疗诊疗聊天」：会话表 + 消息表（含意图标签与引用 JSON）。
dev SQLite 环境下 create_all() 也会自动建表；本迁移面向 MySQL 生产库。
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1c2f3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '5d10b4e3689f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('patient_chat_session',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('session_key', sa.String(length=64), nullable=False, comment='客户端生成/后端签发的会话标识'),
    sa.Column('message_count', sa.Integer(), nullable=False, comment='累计消息数'),
    sa.Column('client_ip', sa.String(length=50), nullable=False, comment='最近来源 IP（限流审计）'),
    sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
    sa.Column('updated_at', sa.DateTime(), nullable=False, comment='更新时间'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_key')
    )
    op.create_index('ix_patient_chat_session_session_key', 'patient_chat_session', ['session_key'])

    op.create_table('patient_chat_message',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('session_id', sa.Integer(), nullable=False, comment='会话 ID'),
    sa.Column('role', sa.String(length=10), nullable=False, comment='user|assistant'),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('intent', sa.String(length=20), nullable=False, comment='chitchat|consult|inquiry|emergency（仅 user 消息）'),
    sa.Column('citations', sa.Text(), nullable=False, comment='证据引用 JSON（仅 assistant 消息）'),
    sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
    sa.Column('updated_at', sa.DateTime(), nullable=False, comment='更新时间'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_patient_chat_message_session_id', 'patient_chat_message', ['session_id'])


def downgrade() -> None:
    op.drop_index('ix_patient_chat_message_session_id', table_name='patient_chat_message')
    op.drop_table('patient_chat_message')
    op.drop_index('ix_patient_chat_session_session_key', table_name='patient_chat_session')
    op.drop_table('patient_chat_session')
