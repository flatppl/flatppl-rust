module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %arg1, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %7 = stablehlo.subtract %6, %arg0 : tensor<f32>
    %8 = stablehlo.divide %7, %arg1 : tensor<f32>
    %9 = stablehlo.atan2 %8, %0 : tensor<f32>
    %10 = stablehlo.constant dense<0.3183098861837907> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.constant dense<0.5> : tensor<f32>
    %13 = stablehlo.add %12, %11 : tensor<f32>
    %14 = stablehlo.subtract %0, %13 : tensor<f32>
    %15 = stablehlo.compare EQ, %14, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = stablehlo.select %15, %0, %14 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.log %17 : tensor<f32>
    %19 = stablehlo.select %15, %16, %18 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.abs %19 : tensor<f32>
    %21 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %22 = stablehlo.compare LT, %20, %21 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %24 = stablehlo.constant dense<-1.1447298858494002> : tensor<f32>
    %25 = stablehlo.log %arg1 : tensor<f32>
    %26 = stablehlo.negate %25 : tensor<f32>
    %27 = stablehlo.subtract %12, %arg0 : tensor<f32>
    %28 = stablehlo.divide %27, %arg1 : tensor<f32>
    %29 = stablehlo.multiply %28, %28 : tensor<f32>
    %30 = stablehlo.add %0, %29 : tensor<f32>
    %31 = stablehlo.log %30 : tensor<f32>
    %32 = stablehlo.negate %31 : tensor<f32>
    %33 = stablehlo.add %24, %26 : tensor<f32>
    %34 = stablehlo.add %33, %32 : tensor<f32>
    %36 = stablehlo.subtract %34, %19 : tensor<f32>
    %37 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %38 = stablehlo.select %22, %36, %37 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %38, %5 : tensor<f32>, tensor<i32>
  }
}
