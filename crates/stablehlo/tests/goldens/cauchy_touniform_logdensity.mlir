module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %4 = stablehlo.subtract %0, %3 : tensor<f32>
    %5 = stablehlo.subtract %2, %0 : tensor<f32>
    %6 = stablehlo.multiply %4, %5 : tensor<f32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.compare GE, %6, %7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = stablehlo.constant dense<1.0> : tensor<f32>
    %10 = stablehlo.select %8, %0, %9 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = stablehlo.constant dense<-1.1447298858494002> : tensor<f32>
    %12 = stablehlo.log %arg1 : tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.subtract %10, %arg0 : tensor<f32>
    %15 = stablehlo.divide %14, %arg1 : tensor<f32>
    %16 = stablehlo.multiply %15, %15 : tensor<f32>
    %17 = stablehlo.constant dense<1.0> : tensor<f32>
    %18 = stablehlo.add %17, %16 : tensor<f32>
    %19 = stablehlo.log %18 : tensor<f32>
    %20 = stablehlo.negate %19 : tensor<f32>
    %21 = stablehlo.add %11, %13 : tensor<f32>
    %22 = stablehlo.add %21, %20 : tensor<f32>
    %23 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %24 = stablehlo.negate %23 : tensor<f32>
    %25 = stablehlo.select %8, %22, %24 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %26 = stablehlo.subtract %2, %arg0 : tensor<f32>
    %27 = stablehlo.divide %26, %arg1 : tensor<f32>
    %28 = stablehlo.constant dense<1.0> : tensor<f32>
    %29 = stablehlo.atan2 %27, %28 : tensor<f32>
    %30 = stablehlo.constant dense<0.3183098861837907> : tensor<f32>
    %31 = stablehlo.multiply %30, %29 : tensor<f32>
    %32 = stablehlo.constant dense<0.5> : tensor<f32>
    %33 = stablehlo.add %32, %31 : tensor<f32>
    %34 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %35 = stablehlo.subtract %34, %arg0 : tensor<f32>
    %36 = stablehlo.divide %35, %arg1 : tensor<f32>
    %37 = stablehlo.constant dense<1.0> : tensor<f32>
    %38 = stablehlo.atan2 %36, %37 : tensor<f32>
    %39 = stablehlo.constant dense<0.3183098861837907> : tensor<f32>
    %40 = stablehlo.multiply %39, %38 : tensor<f32>
    %41 = stablehlo.constant dense<0.5> : tensor<f32>
    %42 = stablehlo.add %41, %40 : tensor<f32>
    %43 = stablehlo.subtract %33, %42 : tensor<f32>
    %44 = stablehlo.log %43 : tensor<f32>
    %45 = stablehlo.subtract %25, %44 : tensor<f32>
    return %45 : tensor<f32>
  }
}
