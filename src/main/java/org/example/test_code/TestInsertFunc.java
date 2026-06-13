package org.example.test_code;

import java.sql.Connection;
import java.sql.SQLException;

public class TestInsertFunc {
    private Connection conn;

    public void executeRaw(String sql) throws SQLException {
        conn.createStatement().execute(sql);
    }
}

class UserService {
    private TestInsertFunc db;

    public void updateUserInfo(String userId, String bio) throws SQLException {
        String command = "UPDATE profiles SET bio = '" + bio + "' WHERE id = " + userId;

        db.executeRaw(command);
    }
}
